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
    greedy_multi_tree_ub_seeded,
};
pub use red_blue::red_blue_approx;

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
        return MafBounds { lower: 1, upper: 1, best_partition: None };
    }

    let m = trees.len();
    let mut best_lb = 1usize;
    let mut best_ub_pair = usize::MAX;

    // lb_cuts[i][j] = best known lower bound on the pairwise cuts d(Ti, Tj).
    let mut lb_cuts = vec![vec![0usize; m]; m];

    for i in 0..m {
        for j in (i + 1)..m {
            let approx_2 = red_blue_approx(&trees[i], &trees[j]);
            let pair_lb = approx_2.div_ceil(2);
            lb_cuts[i][j] = pair_lb;
            lb_cuts[j][i] = pair_lb;

            let lb_components = pair_lb + 1;
            if lb_components > best_lb {
                best_lb = lb_components;
            }
            let approx_ub_components = approx_2 + 1;
            if approx_ub_components < best_ub_pair {
                best_ub_pair = approx_ub_components;
            }

            let cherry_ub = cherry_reduce_ub(&trees[i], &trees[j]);
            let ub_components = cherry_ub + 1;
            if ub_components < best_ub_pair {
                best_ub_pair = ub_components;
            }
        }
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

    let (upper, best_partition) = if trees.len() == 2 {
        (best_ub_pair.min(num_leaves as usize), None)
    } else {
        // Multi-tree: run greedy with 20 seeds per ref_idx (matching the 2-tree cherry_reduce_ub).
        // Each run is O(n^2) and takes <1ms for n≤200, so 20*m runs is cheap.
        // Track the best (ref_idx, seed) pair to return the best partition.
        let mut best_multi_ub = num_leaves as usize;
        let mut best_ref = 0usize;
        let mut best_seed = 0u64;
        for ref_idx in 0..m {
            for seed in 0..=20u64 {
                let ub = cherry::greedy_multi_tree_ub_seeded(trees, ref_idx, seed);
                if ub < best_multi_ub {
                    best_multi_ub = ub;
                    best_ref = ref_idx;
                    best_seed = seed;
                }
            }
        }
        let ub = best_multi_ub.min(num_leaves as usize);
        let (_, partition) = cherry::greedy_multi_tree_partition(trees, best_ref, best_seed);
        (ub, Some(partition))
    };

    MafBounds {
        lower: best_lb.min(upper),
        upper,
        best_partition,
    }
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
