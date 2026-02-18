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

use klados_core::tree::Tree;

pub use cherry::cherry_reduce_ub;
pub use red_blue::red_blue_approx;

pub struct MafBounds {
    pub lower: usize,
    pub upper: usize,
}

pub fn maf_bounds(trees: &[Tree], num_leaves: u32) -> MafBounds {
    if trees.len() <= 1 {
        return MafBounds { lower: 1, upper: 1 };
    }

    let m = trees.len();
    let mut best_lb = 1usize;
    let mut best_ub_pair = usize::MAX;

    for i in 0..m {
        for j in (i + 1)..m {
            let approx_2 = red_blue_approx(&trees[i], &trees[j]);
            let lb_cost = (approx_2 + 1) / 2;
            let lb_components = lb_cost + 1;
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

    let upper = if trees.len() == 2 {
        best_ub_pair.min(num_leaves as usize)
    } else {
        let mut best_multi_ub = num_leaves as usize;
        for ref_idx in 0..m {
            let ub = cherry::greedy_multi_tree_ub(trees, ref_idx);
            if ub < best_multi_ub {
                best_multi_ub = ub;
            }
        }
        best_multi_ub.min(num_leaves as usize)
    };

    MafBounds {
        lower: best_lb.min(upper),
        upper,
    }
}

pub fn lower_bound_components(trees: &[Tree]) -> usize {
    if trees.len() <= 1 {
        return 1;
    }
    maf_bounds(trees, trees[0].num_leaves).lower
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
