use std::fs;
use std::io::{BufReader, IsTerminal};
use std::path::PathBuf;
use std::time::Instant;

use clap::ValueEnum;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use klados_core::{Instance, Tree};
use klados_exact::whidden::{WhiddenProgressUpdate, WhiddenRuleStats, WhiddenSolver};
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;
use serde::Serialize;

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    Human,
    Json,
    Ndjson,
}

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq)]
pub enum ProgressMode {
    Auto,
    On,
    Off,
}

#[derive(Clone, Debug, Default, Serialize)]
struct InstanceStatsRow {
    digest: String,
    num_trees: usize,
    num_leaves: u32,
    solved: bool,
    score: Option<usize>,
    elapsed_ms: f64,
    nodes_explored: u64,
    k_attempts: u64,
    k_success: Option<usize>,
    k_total_elapsed_ms: f64,
    k_last_elapsed_ms: f64,
    rules: RuleSnapshot,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RuleSnapshot {
    rule_cob_fired: u64,
    rule_rcob_a_fired: u64,
    rule_rcob_c_fired: u64,
    rule_cut_two_b_fired: u64,
    rule_cut_all_b_forced: u64,
    rule_mestel6_forced: u64,
    prefer_nonbranching_hits: u64,
    branch_a_attempts: u64,
    branch_a_successes: u64,
    branch_b_attempts: u64,
    branch_b_successes: u64,
    branch_c_attempts: u64,
    branch_c_successes: u64,
    prune_k_exhausted: u64,
    prune_bb_approx: u64,
    prune_no_enabled_branches: u64,
    tt_lookups: u64,
    tt_hits: u64,
    tt_prunes: u64,
    tt_stores: u64,
    tt_overwrites: u64,
    mestel6_checks: u64,
    bc_lookups: u64,
    bc_hits: u64,
    bc_stores: u64,
    bb_skipped_by_parent: u64,
    bb_approx3_calls: u64,
    bb_approx2_calls: u64,
    bb_approx2_prunes: u64,
    skip_a_cob: u64,
    skip_a_rcob_c: u64,
    skip_a_ep_protected: u64,
    skip_b_rcob_a: u64,
    skip_b_rcob_c: u64,
    skip_b_separate_components: u64,
    skip_b_ep_protected: u64,
    skip_c_cob: u64,
    skip_c_rcob_a: u64,
    skip_c_ep_protected: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
struct AggregateStats {
    processed: usize,
    solved: usize,
    errors: usize,
    total_elapsed_ms: f64,
    total_nodes_explored: u64,
    total_k_elapsed_ms: f64,
    totals: RuleSnapshot,
}

#[derive(Clone, Debug, Serialize)]
struct JsonReport {
    list_file: String,
    aggregate: AggregateStats,
    instances: Vec<InstanceStatsRow>,
}

pub fn run(
    list_file: &PathBuf,
    max_instances: Option<usize>,
    show_instances: bool,
    format: OutputFormat,
    progress: ProgressMode,
    log_interval_ms: u64,
    output: Option<PathBuf>,
    bb: bool,
    bb_2approx: bool,
    tt_enabled: bool,
    tt_prune: bool,
    tt_size_log2: u8,
    mestel_rule6: bool,
    bound_cache_enabled: bool,
    bound_cache_size_log2: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(list_file)?;
    let lines: Vec<&str> = content.lines().collect();
    let base_dir = list_file.parent().unwrap_or(std::path::Path::new("."));

    let mut rows: Vec<InstanceStatsRow> = Vec::new();
    let mut agg = AggregateStats::default();

    let total_candidates = lines
        .iter()
        .filter(|line| {
            let t = line.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .count();
    let total_target = max_instances
        .map(|m| m.min(total_candidates))
        .unwrap_or(total_candidates);

    let is_tty = std::io::stderr().is_terminal();
    let line_logging = progress == ProgressMode::On && !is_tty;
    let progress_enabled = match progress {
        ProgressMode::On => is_tty,
        ProgressMode::Off => false,
        ProgressMode::Auto => is_tty,
    };

    let progress_bar = if progress_enabled {
        let pb = ProgressBar::new(total_target as u64);
        pb.set_draw_target(ProgressDrawTarget::stderr());
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} {pos:>4}/{len:4} {wide_bar:.cyan/blue} {percent:>3}% | {elapsed_precise}<{eta_precise} | {msg}",
            )?
            .progress_chars("=> "),
        );
        Some(pb)
    } else {
        None
    };

    let interval = log_interval_ms.max(50) as f64;

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(max) = max_instances {
            if agg.processed >= max {
                break;
            }
        }

        let digest = line.strip_prefix("s:").unwrap_or(line).to_string();
        let rel = format!(
            "stride-downloads/{}/{}/{}",
            &digest[..2],
            &digest[2..4],
            &digest[4..]
        );
        let path = {
            let p = base_dir.join(&rel);
            if p.exists() {
                p
            } else {
                let p2 = list_file
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .join(&rel);
                if p2.exists() {
                    p2
                } else {
                    let p3 = std::path::Path::new(".").join(&rel);
                    if p3.exists() {
                        p3
                    } else {
                        std::path::Path::new("..").join(&rel)
                    }
                }
            }
        };

        if !path.exists() {
            agg.errors += 1;
            agg.processed += 1;
            if let Some(pb) = &progress_bar {
                pb.inc(1);
                pb.set_message(format!("{} missing", short_digest(&digest)));
            }
            continue;
        }

        let file_content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                agg.errors += 1;
                agg.processed += 1;
                if let Some(pb) = &progress_bar {
                    pb.inc(1);
                    pb.set_message(format!("{} read error", short_digest(&digest)));
                }
                continue;
            }
        };

        let reader = BufReader::new(file_content.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
                agg.errors += 1;
                agg.processed += 1;
                if let Some(pb) = &progress_bar {
                    pb.inc(1);
                    pb.set_message(format!("{} parse error", short_digest(&digest)));
                }
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

        let mut solver = WhiddenSolver::new()
            .with_bb(bb)
            .with_bb_2approx(bb_2approx)
            .with_tt_enabled(tt_enabled)
            .with_tt_prune(tt_prune)
            .with_tt_size_log2(tt_size_log2)
            .with_mestel_rule6(mestel_rule6)
            .with_bound_cache_enabled(bound_cache_enabled)
            .with_bound_cache_size_log2(bound_cache_size_log2);
        let start = Instant::now();
        let mut last_log_mark = 0.0_f64;
        let digest_short = digest[..16.min(digest.len())].to_string();
        let mut callback = |u: WhiddenProgressUpdate| {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if let Some(pb) = &progress_bar {
                pb.set_message(format!(
                    "{} | {} | try {:>3} | nodes {:>7} | k-step {:>8}{}",
                    short_digest(&digest_short),
                    fmt_k_range(u.current_k, u.lb_k, u.ub_k),
                    u.k_attempts,
                    fmt_nodes(u.nodes_explored),
                    fmt_ms(u.k_elapsed_ms),
                    if u.solved { " | solved" } else { "" }
                ));
            } else if line_logging && elapsed - last_log_mark >= interval {
                eprintln!(
                    "[whidden-stats] {}/{} {} | {} | try {:>3} | nodes {:>7} | k-step {:>8}",
                    agg.processed + 1,
                    total_target,
                    short_digest(&digest_short),
                    fmt_k_range(u.current_k, u.lb_k, u.ub_k),
                    u.k_attempts,
                    fmt_nodes(u.nodes_explored),
                    fmt_ms(u.k_elapsed_ms),
                );
                last_log_mark = elapsed;
            }
        };

        let solved = solver.solve_with_progress(&instance, Some(&mut callback));
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let score = solved.as_ref().map(|v| v.len());
        let solved_flag = solved.is_some();

        let rs: &WhiddenRuleStats = solver.rule_stats();
        let row = InstanceStatsRow {
            digest: digest.clone(),
            num_trees: instance.num_trees(),
            num_leaves: instance.num_leaves,
            solved: solved_flag,
            score,
            elapsed_ms,
            nodes_explored: solver.solver_stats().nodes_explored,
            k_attempts: rs.k_attempts,
            k_success: rs.k_success,
            k_total_elapsed_ms: rs.k_total_elapsed_ms,
            k_last_elapsed_ms: rs.k_last_elapsed_ms,
            rules: RuleSnapshot {
                rule_cob_fired: rs.rule_cob_fired,
                rule_rcob_a_fired: rs.rule_rcob_a_fired,
                rule_rcob_c_fired: rs.rule_rcob_c_fired,
                rule_cut_two_b_fired: rs.rule_cut_two_b_fired,
                rule_cut_all_b_forced: rs.rule_cut_all_b_forced,
                rule_mestel6_forced: rs.rule_mestel6_forced,
                prefer_nonbranching_hits: rs.prefer_nonbranching_hits,
                branch_a_attempts: rs.branch_a_attempts,
                branch_a_successes: rs.branch_a_successes,
                branch_b_attempts: rs.branch_b_attempts,
                branch_b_successes: rs.branch_b_successes,
                branch_c_attempts: rs.branch_c_attempts,
                branch_c_successes: rs.branch_c_successes,
                prune_k_exhausted: rs.prune_k_exhausted,
                prune_bb_approx: rs.prune_bb_approx,
                prune_no_enabled_branches: rs.prune_no_enabled_branches,
                tt_lookups: rs.tt_lookups,
                tt_hits: rs.tt_hits,
                tt_prunes: rs.tt_prunes,
                tt_stores: rs.tt_stores,
                tt_overwrites: rs.tt_overwrites,
                mestel6_checks: rs.mestel6_checks,
                bc_lookups: rs.bc_lookups,
                bc_hits: rs.bc_hits,
                bc_stores: rs.bc_stores,
                bb_skipped_by_parent: rs.bb_skipped_by_parent,
                bb_approx3_calls: rs.bb_approx3_calls,
                bb_approx2_calls: rs.bb_approx2_calls,
                bb_approx2_prunes: rs.bb_approx2_prunes,
                skip_a_cob: rs.skip_a_cob,
                skip_a_rcob_c: rs.skip_a_rcob_c,
                skip_a_ep_protected: rs.skip_a_ep_protected,
                skip_b_rcob_a: rs.skip_b_rcob_a,
                skip_b_rcob_c: rs.skip_b_rcob_c,
                skip_b_separate_components: rs.skip_b_separate_components,
                skip_b_ep_protected: rs.skip_b_ep_protected,
                skip_c_cob: rs.skip_c_cob,
                skip_c_rcob_a: rs.skip_c_rcob_a,
                skip_c_ep_protected: rs.skip_c_ep_protected,
            },
        };

        agg.processed += 1;
        if solved_flag {
            agg.solved += 1;
        } else {
            agg.errors += 1;
        }
        agg.total_elapsed_ms += elapsed_ms;
        agg.total_nodes_explored += row.nodes_explored;
        agg.total_k_elapsed_ms += row.k_total_elapsed_ms;

        agg.totals.rule_cob_fired += row.rules.rule_cob_fired;
        agg.totals.rule_rcob_a_fired += row.rules.rule_rcob_a_fired;
        agg.totals.rule_rcob_c_fired += row.rules.rule_rcob_c_fired;
        agg.totals.rule_cut_two_b_fired += row.rules.rule_cut_two_b_fired;
        agg.totals.rule_cut_all_b_forced += row.rules.rule_cut_all_b_forced;
        agg.totals.rule_mestel6_forced += row.rules.rule_mestel6_forced;
        agg.totals.prefer_nonbranching_hits += row.rules.prefer_nonbranching_hits;
        agg.totals.branch_a_attempts += row.rules.branch_a_attempts;
        agg.totals.branch_a_successes += row.rules.branch_a_successes;
        agg.totals.branch_b_attempts += row.rules.branch_b_attempts;
        agg.totals.branch_b_successes += row.rules.branch_b_successes;
        agg.totals.branch_c_attempts += row.rules.branch_c_attempts;
        agg.totals.branch_c_successes += row.rules.branch_c_successes;
        agg.totals.prune_k_exhausted += row.rules.prune_k_exhausted;
        agg.totals.prune_bb_approx += row.rules.prune_bb_approx;
        agg.totals.prune_no_enabled_branches += row.rules.prune_no_enabled_branches;
        agg.totals.tt_lookups += row.rules.tt_lookups;
        agg.totals.tt_hits += row.rules.tt_hits;
        agg.totals.tt_prunes += row.rules.tt_prunes;
        agg.totals.tt_stores += row.rules.tt_stores;
        agg.totals.tt_overwrites += row.rules.tt_overwrites;
        agg.totals.mestel6_checks += row.rules.mestel6_checks;
        agg.totals.bc_lookups += row.rules.bc_lookups;
        agg.totals.bc_hits += row.rules.bc_hits;
        agg.totals.bc_stores += row.rules.bc_stores;
        agg.totals.bb_skipped_by_parent += row.rules.bb_skipped_by_parent;
        agg.totals.bb_approx3_calls += row.rules.bb_approx3_calls;
        agg.totals.bb_approx2_calls += row.rules.bb_approx2_calls;
        agg.totals.bb_approx2_prunes += row.rules.bb_approx2_prunes;
        agg.totals.skip_a_cob += row.rules.skip_a_cob;
        agg.totals.skip_a_rcob_c += row.rules.skip_a_rcob_c;
        agg.totals.skip_a_ep_protected += row.rules.skip_a_ep_protected;
        agg.totals.skip_b_rcob_a += row.rules.skip_b_rcob_a;
        agg.totals.skip_b_rcob_c += row.rules.skip_b_rcob_c;
        agg.totals.skip_b_separate_components += row.rules.skip_b_separate_components;
        agg.totals.skip_b_ep_protected += row.rules.skip_b_ep_protected;
        agg.totals.skip_c_cob += row.rules.skip_c_cob;
        agg.totals.skip_c_rcob_a += row.rules.skip_c_rcob_a;
        agg.totals.skip_c_ep_protected += row.rules.skip_c_ep_protected;

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message(format!(
                "{} | done | score {:>4} | elapsed {:>8} | nodes {:>7}",
                short_digest(&row.digest),
                row.score
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "ERR".to_string()),
                fmt_ms(row.elapsed_ms),
                fmt_nodes(row.nodes_explored),
            ));
        }

        rows.push(row);
    }

    if let Some(pb) = &progress_bar {
        pb.finish_and_clear();
    }

    let render = match format {
        OutputFormat::Human => render_human(&agg, &rows, show_instances),
        OutputFormat::Json => serde_json::to_string_pretty(&JsonReport {
            list_file: list_file.display().to_string(),
            aggregate: agg.clone(),
            instances: rows.clone(),
        })?,
        OutputFormat::Ndjson => {
            let mut lines = Vec::new();
            for row in &rows {
                lines.push(serde_json::to_string(row)?);
            }
            lines.push(serde_json::to_string(&serde_json::json!({
                "type": "aggregate",
                "list_file": list_file.display().to_string(),
                "aggregate": agg,
            }))?);
            lines.join("\n")
        }
    };

    if let Some(out) = output {
        fs::write(out, render)?;
    } else {
        println!("{}", render);
    }

    Ok(())
}

fn short_digest(digest: &str) -> String {
    if digest.len() <= 12 {
        return digest.to_string();
    }
    format!("{}..{}", &digest[..6], &digest[digest.len() - 4..])
}

fn fmt_k_range(current: usize, lb: usize, ub: usize) -> String {
    format!("k {:>3} [{:>3}..{:>3}]", current, lb, ub)
}

fn fmt_nodes(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.1}ms", ms)
    }
}

fn pct(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * (num as f64) / (den as f64)
    }
}

fn pct_str(num: u64, den: u64) -> String {
    if den == 0 {
        "0.00%".to_string()
    } else {
        let p = pct(num, den);
        if p >= 0.1 {
            format!("{:.2}%", p)
        } else {
            format!("{:.4}%", p)
        }
    }
}

fn rate_str(success: u64, attempts: u64) -> String {
    format!("{}/{} ({})", success, attempts, pct_str(success, attempts))
}

fn render_human(agg: &AggregateStats, rows: &[InstanceStatsRow], show_instances: bool) -> String {
    let mut out = String::new();
    out.push_str("=== Whidden Rule Statistics ===\n\n");
    out.push_str(&format!(
        "processed={} solved={} errors={} total_ms={:.1} avg_ms={:.1} avg_nodes={:.1}\n",
        agg.processed,
        agg.solved,
        agg.errors,
        agg.total_elapsed_ms,
        if agg.processed > 0 {
            agg.total_elapsed_ms / agg.processed as f64
        } else {
            0.0
        },
        if agg.processed > 0 {
            agg.total_nodes_explored as f64 / agg.processed as f64
        } else {
            0.0
        },
    ));
    out.push_str(&format!(
        "k_total_ms={:.1} avg_k_ms_per_instance={:.1}\n\n",
        agg.total_k_elapsed_ms,
        if agg.processed > 0 {
            agg.total_k_elapsed_ms / agg.processed as f64
        } else {
            0.0
        },
    ));

    let mut rule_ranking: Vec<(&str, u64)> = vec![
        ("CUT_ALL_B forced", agg.totals.rule_cut_all_b_forced),
        ("MESTEL6 forced", agg.totals.rule_mestel6_forced),
        ("COB", agg.totals.rule_cob_fired),
        ("CUT_TWO_B", agg.totals.rule_cut_two_b_fired),
        ("Prefer nonbranching", agg.totals.prefer_nonbranching_hits),
        ("RCOB-C", agg.totals.rule_rcob_c_fired),
        ("RCOB-A", agg.totals.rule_rcob_a_fired),
    ];
    rule_ranking.sort_by(|a, b| b.1.cmp(&a.1));
    let rule_total: u64 = rule_ranking.iter().map(|(_, c)| *c).sum();

    out.push_str("-- Rule Fire Ranking --\n");
    out.push_str(&format!(
        "{:<3} {:<20} {:>14} {:>10}\n",
        "#", "rule", "count", "share"
    ));
    out.push_str(&format!("{}\n", "-".repeat(54)));
    for (idx, (name, count)) in rule_ranking.iter().enumerate() {
        out.push_str(&format!(
            "{:<3} {:<20} {:>14} {:>10}\n",
            idx + 1,
            name,
            count,
            pct_str(*count, rule_total),
        ));
    }
    out.push_str("\n");

    let attempts_total =
        agg.totals.branch_a_attempts + agg.totals.branch_b_attempts + agg.totals.branch_c_attempts;
    out.push_str("-- Branch Outcomes --\n");
    out.push_str(&format!(
        "A: {} | share of attempts={}\n",
        rate_str(agg.totals.branch_a_successes, agg.totals.branch_a_attempts),
        pct_str(agg.totals.branch_a_attempts, attempts_total)
    ));
    out.push_str(&format!(
        "B: {} | share of attempts={}\n",
        rate_str(agg.totals.branch_b_successes, agg.totals.branch_b_attempts),
        pct_str(agg.totals.branch_b_attempts, attempts_total)
    ));
    out.push_str(&format!(
        "C: {} | share of attempts={}\n\n",
        rate_str(agg.totals.branch_c_successes, agg.totals.branch_c_attempts),
        pct_str(agg.totals.branch_c_attempts, attempts_total)
    ));

    let mut prune_ranking: Vec<(&str, u64)> = vec![
        ("BB approx prune", agg.totals.prune_bb_approx),
        ("no enabled branches", agg.totals.prune_no_enabled_branches),
        ("k exhausted", agg.totals.prune_k_exhausted),
    ];
    prune_ranking.sort_by(|a, b| b.1.cmp(&a.1));
    let prune_total: u64 = prune_ranking.iter().map(|(_, c)| *c).sum();

    out.push_str("-- Pruning Ranking --\n");
    out.push_str(&format!(
        "{:<3} {:<20} {:>14} {:>10}\n",
        "#", "reason", "count", "share"
    ));
    out.push_str(&format!("{}\n", "-".repeat(54)));
    for (idx, (name, count)) in prune_ranking.iter().enumerate() {
        out.push_str(&format!(
            "{:<3} {:<20} {:>14} {:>10}\n",
            idx + 1,
            name,
            count,
            pct_str(*count, prune_total),
        ));
    }
    out.push_str("\n");

    out.push_str("-- Transposition Table --\n");
    out.push_str(&format!(
        "lookups={} hits={} ({}) prunes={} stores={} overwrites={}\n\n",
        agg.totals.tt_lookups,
        agg.totals.tt_hits,
        pct_str(agg.totals.tt_hits, agg.totals.tt_lookups),
        agg.totals.tt_prunes,
        agg.totals.tt_stores,
        agg.totals.tt_overwrites,
    ));

    out.push_str("-- Bound Cache & Propagation --\n");
    out.push_str(&format!(
        "mestel6: checks={} forced={}\n",
        agg.totals.mestel6_checks, agg.totals.rule_mestel6_forced,
    ));
    out.push_str(&format!(
        "bc: lookups={} hits={} ({}) stores={}\n",
        agg.totals.bc_lookups,
        agg.totals.bc_hits,
        pct_str(agg.totals.bc_hits, agg.totals.bc_lookups),
        agg.totals.bc_stores,
    ));
    out.push_str(&format!(
        "bb: skipped_by_parent={} approx3_calls={} approx2_calls={} approx2_prunes={}\n\n",
        agg.totals.bb_skipped_by_parent,
        agg.totals.bb_approx3_calls,
        agg.totals.bb_approx2_calls,
        agg.totals.bb_approx2_prunes,
    ));

    let mut skip_ranking: Vec<(&str, u64)> = vec![
        ("skip_a_cob", agg.totals.skip_a_cob),
        ("skip_a_rcob_c", agg.totals.skip_a_rcob_c),
        ("skip_a_ep", agg.totals.skip_a_ep_protected),
        ("skip_b_rcob_a", agg.totals.skip_b_rcob_a),
        ("skip_b_rcob_c", agg.totals.skip_b_rcob_c),
        ("skip_b_sep_comp", agg.totals.skip_b_separate_components),
        ("skip_b_ep", agg.totals.skip_b_ep_protected),
        ("skip_c_cob", agg.totals.skip_c_cob),
        ("skip_c_rcob_a", agg.totals.skip_c_rcob_a),
        ("skip_c_ep", agg.totals.skip_c_ep_protected),
    ];
    skip_ranking.sort_by(|a, b| b.1.cmp(&a.1));
    let skip_total: u64 = skip_ranking.iter().map(|(_, c)| *c).sum();

    out.push_str("-- Branch Skip Reasons --\n");
    out.push_str(&format!(
        "{:<3} {:<20} {:>14} {:>10}\n",
        "#", "reason", "count", "share"
    ));
    out.push_str(&format!("{}\n", "-".repeat(54)));
    for (idx, (name, count)) in skip_ranking.iter().enumerate() {
        out.push_str(&format!(
            "{:<3} {:<20} {:>14} {:>10}\n",
            idx + 1,
            name,
            count,
            pct_str(*count, skip_total),
        ));
    }
    out.push_str("\n");

    if show_instances {
        out.push_str("-- Instances --\n");
        out.push_str(&format!(
            "{:<16} {:>3} {:>6} {:>6} {:>8} {:>10} {:>6} {:>8}\n",
            "digest", "m", "n", "score", "ms", "nodes", "k_try", "k_ms"
        ));
        out.push_str(&format!("{}\n", "-".repeat(88)));
        for row in rows {
            out.push_str(&format!(
                "{:<16} {:>3} {:>6} {:>6} {:>8.1} {:>10} {:>6} {:>8.1}\n",
                &row.digest[..16.min(row.digest.len())],
                row.num_trees,
                row.num_leaves,
                row.score
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "ERR".to_string()),
                row.elapsed_ms,
                row.nodes_explored,
                row.k_attempts,
                row.k_total_elapsed_ms,
            ));
        }
    }

    out
}
