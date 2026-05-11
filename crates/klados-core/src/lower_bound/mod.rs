//! Bound computation for MAF via the Red-Blue 2-approximation algorithm
//! (Olver, Schalekamp, van der Ster, Stougie, van Zuylen, 2018) for lower
//! bounds, and cherry-based heuristic for upper bounds.
//!
//! The 2-approximation guarantees: true_dist <= approx_cost <= 2 * true_dist.
//! This gives a valid lower bound: OPT >= ceil(approx_cost / 2).

mod cherry;
mod feasibility;
mod partition;
mod red_blue;
mod tree_data;

use crate::tree::Tree;

pub use cherry::{
    cherry_reduce_ub, greedy_multi_tree_partition, greedy_multi_tree_ub,
    greedy_multi_tree_ub_seeded, pairwise_refine_ub,
};
pub use red_blue::{RedBlueResult, red_blue_approx, red_blue_approx_detailed};

pub struct MafBounds {
    pub lower: usize,
    pub upper: usize,
    /// The best greedy partition found during UB computation.
    ///
    /// `partition[j]` is the 0-based component index for the leaf with label `j+1`.
    /// Suitable for use as SAT phase hints or a warm-start solution.
    /// `None` for trivial (single-tree) instances.
    pub best_partition: Option<Vec<usize>>,
}

pub fn maf_bounds(trees: &[Tree], num_leaves: u32) -> MafBounds {
    if trees.len() <= 1 {
        return MafBounds {
            lower: 1,
            upper: 1,
            best_partition: None,
        };
    }

    let m = trees.len();
    let mut best_lb = 1usize;
    let mut best_ub_pair = usize::MAX;

    // lb_cuts[i][j] = best known lower bound on the pairwise cuts d(Ti, Tj).
    let mut lb_cuts = vec![vec![0usize; m]; m];

    let t_pairwise = std::time::Instant::now();
    let mut t_rb_total_ms = 0.0f64;
    let mut t_cherry_total_ms = 0.0f64;
    let mut rb_calls: usize = 0;
    let mut rb_skipped: usize = 0;

    if m == 2 {
        // Single pair: need both rb (for UB on 2-tree case) and cherry. Keep original path.
        let t_rb = std::time::Instant::now();
        let rb = red_blue_approx_detailed(&trees[0], &trees[1]);
        t_rb_total_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
        rb_calls += 1;

        let pair_lb = rb.dual_lb;
        lb_cuts[0][1] = pair_lb;
        lb_cuts[1][0] = pair_lb;
        best_lb = best_lb.max(pair_lb + 1);
        best_ub_pair = best_ub_pair.min(rb.ub + 1);

        let t_ch = std::time::Instant::now();
        let cherry_ub = cherry_reduce_ub(&trees[0], &trees[1]);
        t_cherry_total_ms += t_ch.elapsed().as_secs_f64() * 1000.0;
        best_ub_pair = best_ub_pair.min(cherry_ub + 1);
    } else {
        // Stage 1 (cheap): cherry_reduce_ub for all pairs.
        //   - cherry_ub is an UPPER bound on d(Ti, Tj)
        //   - dual_lb (from rb) is a LOWER bound on d(Ti, Tj)
        //   - So dual_lb ≤ cherry_ub
        //   - Max-pair component LB contribution is at most cherry_ub + 1
        //   - If cherry_ub + 1 ≤ current best_lb, rb on this pair cannot tighten it
        let mut pair_cherry: Vec<(usize, usize, usize)> = Vec::with_capacity(m * (m - 1) / 2);
        for i in 0..m {
            for j in (i + 1)..m {
                let t_ch = std::time::Instant::now();
                let cu = cherry_reduce_ub(&trees[i], &trees[j]);
                t_cherry_total_ms += t_ch.elapsed().as_secs_f64() * 1000.0;
                pair_cherry.push((i, j, cu));
            }
        }

        // Stage 2: sort pairs by cherry_ub desc; run rb only where it can tighten best_lb.
        // For skipped pairs we leave lb_cuts[i][j] = 0 (safe under-approximation of dual_lb,
        // so the additive LB stays valid — possibly weaker, never invalidated).
        pair_cherry.sort_by(|a, b| b.2.cmp(&a.2));
        for &(i, j, cu) in &pair_cherry {
            // best_ub_pair is only used for m==2, so don't bother updating it here.
            if cu + 1 <= best_lb {
                rb_skipped += 1;
                continue;
            }
            let t_rb = std::time::Instant::now();
            let rb = red_blue_approx_detailed(&trees[i], &trees[j]);
            t_rb_total_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
            rb_calls += 1;

            let pair_lb = rb.dual_lb;
            lb_cuts[i][j] = pair_lb;
            lb_cuts[j][i] = pair_lb;
            best_lb = best_lb.max(pair_lb + 1);
        }
    }

    let pairwise_total_ms = t_pairwise.elapsed().as_secs_f64() * 1000.0;
    if m >= 5 {
        eprintln!(
            "[bounds] pairwise m={}: total={:.1}ms (red_blue={:.1}ms/{}calls, cherry_reduce={:.1}ms, rb_skipped={}, best_lb={})",
            m, pairwise_total_ms, t_rb_total_ms, rb_calls, t_cherry_total_ms, rb_skipped, best_lb,
        );
    }

    // Additive multi-tree LB: for each reference Ti,
    //   MAF_size ≥ ceil(sum_{j≠i} d(Ti,Tj) / (m-1)) + 1
    // Uses the red-blue approx LBs — zero extra cost, potentially much tighter than max-pair LB.
    if m >= 3 {
        for i in 0..m {
            let sum_d: usize = (0..m).filter(|&j| j != i).map(|j| lb_cuts[i][j]).sum();
            let additive_lb_cuts = sum_d.div_ceil(m - 1);
            let additive_lb_comps = additive_lb_cuts + 1;
            if additive_lb_comps > best_lb {
                best_lb = additive_lb_comps;
            }
        }
    }

    // TODO: Triplet-based RH lower bound (Wu 2010) needs EXACT pairwise distances,
    // not approximations. With approximate lb_cuts, the bound can exceed true OPT.
    // Implement once we have exact pairwise rSPR computation with timeout fallback.

    let (upper, best_partition) = if trees.len() == 2 {
        (best_ub_pair.min(num_leaves as usize), None)
    } else {
        // Multi-tree UB: take the best of:
        // (a) All-tree cherry reduction (requires all trees to agree on cherries)
        // (b) Pairwise-seeded partition refinement (start from best pair, split for other trees)
        let n = num_leaves as usize;

        // (a) All-tree cherry reduction.
        let t_multi = std::time::Instant::now();
        let mut best_multi_ub = n;
        let mut best_ref = 0usize;
        let mut best_seed = 0u64;
        for ref_idx in 0..m {
            for seed in 0..=20u64 {
                let ub = greedy_multi_tree_ub_seeded(trees, ref_idx, seed);
                if ub < best_multi_ub {
                    best_multi_ub = ub;
                    best_ref = ref_idx;
                    best_seed = seed;
                }
            }
        }
        let multi_ms = t_multi.elapsed().as_secs_f64() * 1000.0;
        if m >= 5 {
            eprintln!(
                "[bounds] greedy_multi_tree_ub_seeded ({}x21): {:.1}ms",
                m, multi_ms
            );
        }

        // (b) Pairwise-refine: start from best pair's partition, refine for all trees.
        let t_pr = std::time::Instant::now();
        let (pr_ub, pr_partition) = pairwise_refine_ub(trees, n);
        eprintln!(
            "[bounds] Pairwise-refine UB: {} (all-tree cherry: {}) in {:.1}ms",
            pr_ub,
            best_multi_ub,
            t_pr.elapsed().as_secs_f64() * 1000.0
        );

        if pr_ub < best_multi_ub {
            (pr_ub.min(n), Some(pr_partition))
        } else {
            let ub = best_multi_ub.min(n);
            let (_, partition) = greedy_multi_tree_partition(trees, best_ref, best_seed);
            (ub, Some(partition))
        }
    };

    MafBounds {
        lower: best_lb.min(upper),
        upper,
        best_partition,
    }
}

use crate::Instance;
use crate::kernelize::{self, KernelizeConfig};

/// Compute a tight lower bound on the multi-tree MAF using exact pairwise distances.
///
/// For each pair (Ti, Tj), kernelizes and calls `solve_pair` to find the exact
/// pairwise MAF size. For m >= 3, applies the additive formula:
///   MAF_size >= ceil(sum_{j!=i} d(Ti,Tj) / (m-1)) + 1  for any reference tree Ti
///
/// `solve_pair` takes a kernelized 2-tree Instance and returns `Some(num_components)`
/// if solved, or `None` if the solver couldn't determine the answer (e.g., timeout).
///
/// Returns a lower bound >= `approx_lb`.
pub fn exact_pairwise_lower_bound(
    trees: &[Tree],
    num_leaves: u32,
    approx_lb: usize,
    upper_bound: usize,
    total_budget: std::time::Duration,
    solve_pair: &mut dyn FnMut(&Instance) -> Option<usize>,
) -> usize {
    let m = trees.len();
    if m < 3 {
        return approx_lb;
    }

    let mut best_lb = approx_lb;

    // Compute pairwise red_blue_approx for initial LB and sorting.
    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            let two_approx = red_blue_approx(&trees[i], &trees[j]);
            pairs.push((i, j, two_approx));
        }
    }
    // Sort by highest approx LB first — most likely to tighten the bound.
    pairs.sort_by(|a, b| b.2.cmp(&a.2));

    let start = std::time::Instant::now();

    let mut exact_dist: Vec<Vec<Option<usize>>> = vec![vec![None; m]; m];
    let mut lb_dist: Vec<Vec<usize>> = vec![vec![0; m]; m];

    for &(i, j, two_approx) in &pairs {
        let lb = two_approx.div_ceil(2);
        lb_dist[i][j] = lb;
        lb_dist[j][i] = lb;
    }

    for &(i, j, _two_approx) in &pairs {
        if start.elapsed() >= total_budget {
            break;
        }

        // Kernelize the pair to shrink the search space.
        let pair_instance = Instance::new(vec![trees[i].clone(), trees[j].clone()], num_leaves);
        let kern_cfg = KernelizeConfig {
            chain32_multi: false,
            ..KernelizeConfig::default()
        };
        let kern = kernelize::kernelize(&pair_instance, &kern_cfg);
        let pair_reduction = kern.param_reduction;

        if kern.instance.num_leaves <= 1 {
            // Pair fully reduced by kernelization alone.
            let exact_cuts = pair_reduction;
            let exact_comps = exact_cuts + 1;
            exact_dist[i][j] = Some(exact_cuts);
            exact_dist[j][i] = Some(exact_cuts);
            if exact_cuts > lb_dist[i][j] {
                lb_dist[i][j] = exact_cuts;
                lb_dist[j][i] = exact_cuts;
            }
            if exact_comps > best_lb {
                best_lb = exact_comps;
            }
            if best_lb >= upper_bound {
                return best_lb;
            }
            continue;
        }

        // Call the solver on the kernelized pair.
        if let Some(num_comps_reduced) = solve_pair(&kern.instance) {
            let exact_comps = num_comps_reduced + pair_reduction;
            let exact_cuts = exact_comps - 1;
            exact_dist[i][j] = Some(exact_cuts);
            exact_dist[j][i] = Some(exact_cuts);
            if exact_cuts > lb_dist[i][j] {
                lb_dist[i][j] = exact_cuts;
                lb_dist[j][i] = exact_cuts;
            }
            if exact_comps > best_lb {
                best_lb = exact_comps;
            }
            if best_lb >= upper_bound {
                return best_lb;
            }
        }
    }

    // Additive multi-tree LB: for each reference Ti,
    //   MAF_size >= ceil(sum_{j!=i} d(Ti,Tj) / (m-1)) + 1
    for i in 0..m {
        let sum_d: usize = (0..m)
            .filter(|&j| j != i)
            .map(|j| exact_dist[i][j].unwrap_or(lb_dist[i][j]))
            .sum();
        let lb_cuts = sum_d.div_ceil(m - 1);
        let lb_comps = lb_cuts + 1;
        if lb_comps > best_lb {
            best_lb = lb_comps;
        }
    }

    best_lb
}

pub fn approx_rspr_distance_pub(t1: &Tree, t2: &Tree) -> usize {
    cherry_reduce_ub(t1, t2)
}

pub fn red_blue_approx_pub(t1: &Tree, t2: &Tree) -> usize {
    let result = red_blue_approx(t1, t2);
    eprintln!(
        "Red-Blue approx for 2 trees: {} (MAF size = {})",
        result,
        result + 1
    );
    result
}

#[cfg(test)]
mod tests;
