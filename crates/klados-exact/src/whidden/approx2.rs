//! Olver et al. 2-approximation dual lower bound for MAF on TwinForest.
//!
//! Direct ROI detection with parent-walk LCA (no sparse table), O(depth)
//! V-set intersection via walk-down.
//!
//! Returns D = |P| - 1 - y_decrements, a valid lower bound on OPT.

use super::forest::{TwinForest, T1, T2};
use klados_core::tree::{Label, NodeId, NONE};

// ---------------------------------------------------------------------------
// Parent-walk LCA — O(depth) per query, zero setup
// ---------------------------------------------------------------------------

#[inline]
fn pw_lca(tf: &TwinForest, ti: usize, depth: &[u16], mut a: NodeId, mut b: NodeId) -> NodeId {
    if a == NONE || b == NONE { return NONE; }
    let mut da = depth[a as usize];
    let mut db = depth[b as usize];
    while da > db { a = tf.parent[ti][a as usize]; da -= 1; }
    while db > da { b = tf.parent[ti][b as usize]; db -= 1; }
    while a != b {
        if a == NONE || b == NONE { return NONE; }
        a = tf.parent[ti][a as usize];
        b = tf.parent[ti][b as usize];
    }
    a
}

/// LCA of all labels in a bitmask.
#[inline]
fn pw_lca_of_mask(tf: &TwinForest, ti: usize, depth: &[u16], mask: u128) -> NodeId {
    let mut result = NONE;
    let mut m = mask;
    while m != 0 {
        let bit = m.trailing_zeros() as u32;
        m &= m - 1;
        let node = tf.label_to_node[ti][(bit + 1) as usize];
        if result == NONE { result = node; }
        else { result = pw_lca(tf, ti, depth, result, node); }
    }
    result
}

// ---------------------------------------------------------------------------
// Partition (bitmask-based for speed)
// ---------------------------------------------------------------------------

struct Partition {
    comp: Vec<u32>,
    masks: Vec<u128>,
    next_id: u32,
}

impl Partition {
    fn new_single(n: u32) -> Self {
        let mut comp = vec![0u32; n as usize + 1];
        let mut mask: u128 = 0;
        for l in 1..=n { comp[l as usize] = 0; mask |= 1u128 << (l - 1); }
        Self { comp, masks: vec![mask], next_id: 1 }
    }

    #[inline] fn count_components(&self) -> usize { self.masks.iter().filter(|&&m| m != 0).count() }

    fn split_off(&mut self, labels_mask: u128) {
        if labels_mask == 0 { return; }
        let first_lbl = labels_mask.trailing_zeros() + 1;
        let old_id = self.comp[first_lbl as usize];
        let new_id = self.next_id;
        self.next_id += 1;
        if new_id as usize >= self.masks.len() {
            self.masks.resize(new_id as usize + 1, 0);
        }
        let mut m = labels_mask;
        while m != 0 {
            let bit = m.trailing_zeros();
            m &= m - 1;
            self.comp[(bit + 1) as usize] = new_id;
        }
        self.masks[old_id as usize] &= !labels_mask;
        self.masks[new_id as usize] = labels_mask;
    }
}

// ---------------------------------------------------------------------------
// Precomputation (leaf masks + depth + post-order in one pass)
// ---------------------------------------------------------------------------

struct TreeInfo {
    post_order: Vec<NodeId>,
    leaf_masks: Vec<u128>,
    depth: Vec<u16>,
}

fn build_tree_info(tf: &TwinForest, ti: usize) -> TreeInfo {
    let n = tf.num_nodes[ti];
    let mut post_order = Vec::with_capacity(n);
    let mut leaf_masks = vec![0u128; n];
    let mut depth = vec![0u16; n];

    for &root in &tf.components[ti] {
        let mut stack: Vec<(NodeId, u16, bool)> = vec![(root, 0, false)];
        while let Some((node, d, visited)) = stack.pop() {
            if node == NONE { continue; }
            if visited {
                post_order.push(node);
                // Propagate leaf masks from children
                let lc = tf.left[ti][node as usize];
                let rc = tf.right[ti][node as usize];
                if lc != NONE { leaf_masks[node as usize] |= leaf_masks[lc as usize]; }
                if rc != NONE { leaf_masks[node as usize] |= leaf_masks[rc as usize]; }
            } else {
                depth[node as usize] = d;
                if tf.is_leaf(ti, node) {
                    let lbl = tf.label[ti][node as usize];
                    if lbl != 0 { leaf_masks[node as usize] = 1u128 << (lbl - 1); }
                    post_order.push(node);
                } else {
                    stack.push((node, d, true));
                    let rc = tf.right[ti][node as usize];
                    let lc = tf.left[ti][node as usize];
                    if rc != NONE { stack.push((rc, d + 1, false)); }
                    if lc != NONE { stack.push((lc, d + 1, false)); }
                }
            }
        }
    }
    TreeInfo { post_order, leaf_masks, depth }
}

// ---------------------------------------------------------------------------
// Find-Lowest-ROI
// ---------------------------------------------------------------------------

/// Check if T1 node u is an ROI for the current partition.
/// `cover` uses generation-counter `gen` to avoid full resets.
fn check_is_roi(
    tf: &TwinForest,
    partition: &Partition,
    t1d: &[u16], t2d: &[u16],
    u_leaves: u128,
    cover_owner: &mut [u32],
    cover_gen: &mut [u32],
    stamp: u32,
    u: NodeId,
    t1_leaf_masks: &[u128],
    t2_leaf_masks: &[u128],
) -> bool {
    let num_comps = partition.masks.len() as u32;

    let u_l = tf.left[T1][u as usize];
    let u_r = tf.right[T1][u as usize];
    let u_l_leaves = if u_l != NONE { t1_leaf_masks[u_l as usize] } else { 0 };
    let u_r_leaves = if u_r != NONE { t1_leaf_masks[u_r as usize] } else { 0 };

    for cid in 0..num_comps {
        let a_mask = partition.masks[cid as usize];
        if a_mask == 0 { continue; }
        let inside = a_mask & u_leaves;
        if inside == 0 { continue; }

        // Condition A: component spans both children of u, and interleaved in T2
        let in_l = a_mask & u_l_leaves;
        let in_r = a_mask & u_r_leaves;
        if in_l != 0 && in_r != 0 {
            let lca_l = pw_lca_of_mask(tf, T2, t2d, in_l);
            let lca_r = pw_lca_of_mask(tf, T2, t2d, in_r);
            if lca_l != NONE && lca_r != NONE {
                let overlap = (t2_leaf_masks[lca_l as usize] & in_r) != 0
                           || (t2_leaf_masks[lca_r as usize] & in_l) != 0;
                if overlap { return true; }
            }
        }

        // Condition C: component extends outside L(u), entire component under lca_T2(inside)
        if inside != a_mask {
            let u_hat = pw_lca_of_mask(tf, T2, t2d, inside);
            if u_hat != NONE {
                if (a_mask & t2_leaf_masks[u_hat as usize]) == a_mask {
                    return true;
                }
            }
        }
    }

    // Condition B: V-set overlap between different components in T1.
    // Uses generation counter to avoid full cover reset.
    for cid in 0..num_comps {
        let mask = partition.masks[cid as usize];
        if mask == 0 { continue; }
        let inside = mask & u_leaves;
        if inside.count_ones() < 2 { continue; }
        let lca_node = pw_lca_of_mask(tf, T1, t1d, inside);
        if lca_node == NONE { continue; }

        let mut m = inside;
        while m != 0 {
            let bit = m.trailing_zeros();
            m &= m - 1;
            let mut cur = tf.label_to_node[T1][(bit as usize) + 1];
            while cur != NONE {
                let ci = cur as usize;
                if cover_gen[ci] == stamp&& cover_owner[ci] == cid { break; }
                if cover_gen[ci] == stamp&& cover_owner[ci] != cid { return true; }
                cover_gen[ci] = stamp;
                cover_owner[ci] = cid;
                if cur == lca_node { break; }
                cur = tf.parent[T1][cur as usize];
            }
        }
    }

    false
}

/// Find lowest (deepest) ROI in T1. Single-pass with early exit.
fn find_lowest_roi(
    tf: &TwinForest,
    partition: &Partition,
    t1d: &[u16], t2d: &[u16],
    t1_post_order: &[NodeId],
    t1_leaf_masks: &[u128],
    t2_leaf_masks: &[u128],
    is_roi_arr: &mut [bool],
    cover_owner: &mut [u32],
    cover_gen: &mut [u32],
    gen_ctr: &mut u32,
) -> Option<NodeId> {
    // Single pass: post-order visits children first, so the first ROI
    // with no ROI children is the lowest.
    for &node in t1_post_order {
        if tf.is_leaf(T1, node) {
            is_roi_arr[node as usize] = false;
            continue;
        }

        *gen_ctr += 1;
        let is_roi = check_is_roi(
            tf, partition, t1d, t2d,
            t1_leaf_masks[node as usize], cover_owner, cover_gen, *gen_ctr,
            node, t1_leaf_masks, t2_leaf_masks,
        );
        is_roi_arr[node as usize] = is_roi;

        if is_roi {
            let lc = tf.left[T1][node as usize];
            let rc = tf.right[T1][node as usize];
            let child_roi = (lc != NONE && !tf.is_leaf(T1, lc) && is_roi_arr[lc as usize])
                         || (rc != NONE && !tf.is_leaf(T1, rc) && is_roi_arr[rc as usize]);
            if !child_roi {
                return Some(node);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Make-R∪B-Compatible
// ---------------------------------------------------------------------------

fn make_rub_compatible(
    tf: &TwinForest,
    partition: &mut Partition,
    t2d: &[u16],
    t2_leaf_masks: &[u128],
    red: u128, blue: u128,
) -> usize {
    let mut num_splits = 0usize;
    loop {
        let mut found = false;
        for cid in 0..partition.masks.len() as u32 {
            let a_mask = partition.masks[cid as usize];
            if a_mask == 0 { continue; }
            let rub = a_mask & (red | blue);
            if rub.count_ones() < 2 { continue; }

            let in_r = a_mask & red;
            let in_b = a_mask & blue;
            if in_r == 0 || in_b == 0 { continue; }

            let lca_r = pw_lca_of_mask(tf, T2, t2d, in_r);
            let lca_b = pw_lca_of_mask(tf, T2, t2d, in_b);
            if lca_r == NONE || lca_b == NONE { continue; }

            let overlap = (t2_leaf_masks[lca_r as usize] & in_b) != 0
                       || (t2_leaf_masks[lca_b as usize] & in_r) != 0;
            if !overlap { continue; }

            // Walk down from lca(R∪B) to find deepest node with both red and blue
            let mut u_hat = pw_lca_of_mask(tf, T2, t2d, rub);
            if u_hat == NONE { continue; }

            loop {
                let lc = tf.left[T2][u_hat as usize];
                let rc = tf.right[T2][u_hat as usize];
                let mut moved = false;
                if lc != NONE {
                    let lc_leaves = t2_leaf_masks[lc as usize] & a_mask;
                    if (lc_leaves & red) != 0 && (lc_leaves & blue) != 0 {
                        u_hat = lc; moved = true;
                    }
                }
                if !moved && rc != NONE {
                    let rc_leaves = t2_leaf_masks[rc as usize] & a_mask;
                    if (rc_leaves & red) != 0 && (rc_leaves & blue) != 0 {
                        u_hat = rc; moved = true;
                    }
                }
                if !moved { break; }
            }

            let split_mask = t2_leaf_masks[u_hat as usize] & a_mask;
            if split_mask != 0 && split_mask != a_mask {
                partition.split_off(split_mask);
                num_splits += 1;
                found = true;
                break;
            }
        }
        if !found { break; }
    }
    num_splits
}

// ---------------------------------------------------------------------------
// Make-Splittable (with O(depth) walk-down for V-set intersection)
// ---------------------------------------------------------------------------

/// Walk down from `start` in T2, finding the deepest node that has both
/// s1 and s2 leaves under it. Explores both children when both qualify
/// to match the reference's "deepest in V-set intersection" behavior.
fn walk_down_deepest(tf: &TwinForest, t2d: &[u16], t2_leaf_masks: &[u128], start: NodeId, s1: u128, s2: u128) -> NodeId {
    let mut best = start;
    let mut best_depth = t2d[start as usize];
    // Fixed-size stack avoids heap allocation; depth <= 128 for bitmask approach.
    let mut stack_buf = [NONE; 128];
    let mut sp = 0usize;
    stack_buf[0] = start;
    sp = 1;
    while sp > 0 {
        sp -= 1;
        let v = stack_buf[sp];
        let lc = tf.left[T2][v as usize];
        let rc = tf.right[T2][v as usize];
        let mut pushed = false;
        if lc != NONE && (t2_leaf_masks[lc as usize] & s1) != 0 && (t2_leaf_masks[lc as usize] & s2) != 0 {
            stack_buf[sp] = lc; sp += 1;
            pushed = true;
        }
        if rc != NONE && (t2_leaf_masks[rc as usize] & s1) != 0 && (t2_leaf_masks[rc as usize] & s2) != 0 {
            stack_buf[sp] = rc; sp += 1;
            pushed = true;
        }
        if !pushed && t2d[v as usize] > best_depth {
            best = v;
            best_depth = t2d[v as usize];
        }
    }
    best
}

fn make_splittable(
    tf: &TwinForest,
    partition: &mut Partition,
    t2d: &[u16],
    t2_leaf_masks: &[u128],
    red: u128, blue: u128, white: u128,
) -> usize {
    let mut num_splits = 0usize;
    loop {
        let mut found = false;
        for cid in 0..partition.masks.len() as u32 {
            let a = partition.masks[cid as usize];
            if a == 0 { continue; }
            let a_r = a & red;
            let a_b = a & blue;
            let a_w = a & white;
            let num_colors = (a_r != 0) as u8 + (a_b != 0) as u8 + (a_w != 0) as u8;
            if num_colors < 2 { continue; }

            let lca_r = pw_lca_of_mask(tf, T2, t2d, a_r);
            let lca_b = pw_lca_of_mask(tf, T2, t2d, a_b);
            let lca_w = pw_lca_of_mask(tf, T2, t2d, a_w);

            // V-set overlap: one LCA must be descendant of the other, with leaves interleaved.
            // Depth constraint is correct here (V-set structural overlap, not compatibility).
            let overlap_rb = a_r.count_ones() >= 2 && a_b.count_ones() >= 2 && lca_r != NONE && lca_b != NONE &&
                (((t2_leaf_masks[lca_r as usize] & a_b) != 0 && t2d[lca_r as usize] >= t2d[lca_b as usize]) ||
                 ((t2_leaf_masks[lca_b as usize] & a_r) != 0 && t2d[lca_b as usize] >= t2d[lca_r as usize]));

            let overlap_rw = a_r.count_ones() >= 2 && a_w.count_ones() >= 2 && lca_r != NONE && lca_w != NONE &&
                (((t2_leaf_masks[lca_r as usize] & a_w) != 0 && t2d[lca_r as usize] >= t2d[lca_w as usize]) ||
                 ((t2_leaf_masks[lca_w as usize] & a_r) != 0 && t2d[lca_w as usize] >= t2d[lca_r as usize]));

            let overlap_bw = a_b.count_ones() >= 2 && a_w.count_ones() >= 2 && lca_b != NONE && lca_w != NONE &&
                (((t2_leaf_masks[lca_b as usize] & a_w) != 0 && t2d[lca_b as usize] >= t2d[lca_w as usize]) ||
                 ((t2_leaf_masks[lca_w as usize] & a_b) != 0 && t2d[lca_w as usize] >= t2d[lca_b as usize]));

            if !overlap_rb && !overlap_rw && !overlap_bw { continue; }

            // Find deepest V-set intersection node via walk-down from the deeper LCA.
            let mut split_node = NONE;

            if overlap_rb {
                let start = if t2d[lca_r as usize] >= t2d[lca_b as usize] { lca_r } else { lca_b };
                split_node = walk_down_deepest(tf, t2d, t2_leaf_masks, start, a_r, a_b);
            }
            if split_node == NONE && overlap_rw {
                let start = if t2d[lca_r as usize] >= t2d[lca_w as usize] { lca_r } else { lca_w };
                split_node = walk_down_deepest(tf, t2d, t2_leaf_masks, start, a_r, a_w);
            }
            if split_node == NONE && overlap_bw {
                let start = if t2d[lca_b as usize] >= t2d[lca_w as usize] { lca_b } else { lca_w };
                split_node = walk_down_deepest(tf, t2d, t2_leaf_masks, start, a_b, a_w);
            }

            if split_node != NONE {
                let under = t2_leaf_masks[split_node as usize] & a;
                if under != 0 && under != a {
                    partition.split_off(under);
                    num_splits += 1;
                    found = true;
                    break;
                }
            }
        }
        if !found { break; }
    }
    num_splits
}

// ---------------------------------------------------------------------------
// Split procedure
// ---------------------------------------------------------------------------

fn split_procedure(
    tf: &TwinForest,
    partition: &mut Partition,
    t2d: &[u16],
    t2_leaf_masks: &[u128],
    red: u128, blue: u128, white: u128,
) -> usize {
    let mut y_decrements = 0usize;

    let mut cid = 0;
    while cid < partition.masks.len() {
        let a = partition.masks[cid];
        if a == 0 { cid += 1; continue; }
        let a_r = a & red;
        let a_b = a & blue;
        let a_w = a & white;
        let num_colors = (a_r != 0) as u8 + (a_b != 0) as u8 + (a_w != 0) as u8;
        if num_colors <= 1 { cid += 1; continue; }

        if num_colors == 2 {
            if a_r != 0 && (a_b != 0 || a_w != 0) {
                partition.split_off(a_r);
            } else if a_b != 0 && a_w != 0 {
                partition.split_off(a_b);
            }
            cid += 1;
            continue;
        }

        // Tricolored — check for special-split via T2 topology
        let rub_lca = pw_lca_of_mask(tf, T2, t2d, a_r | a_b);
        if rub_lca != NONE {
            let under_rub = t2_leaf_masks[rub_lca as usize] & a;
            let w_in = under_rub & a_w;
            let w_out = a_w & !under_rub;

            let all_compat = w_in == 0;
            let has_compat = w_out != 0;

            if has_compat {
                if all_compat {
                    partition.split_off(a_r);
                    cid += 1;
                } else {
                    y_decrements += 1;
                    let a_outside = a & !under_rub;
                    if a_outside != 0 { partition.split_off(a_outside); }

                    let a_inside = partition.masks[cid];
                    let a_r_in = a_inside & red;
                    let a_b_in = a_inside & blue;
                    if a_r_in != 0 { partition.split_off(a_r_in); }
                    if a_b_in != 0 { partition.split_off(a_b_in); }
                    cid += 1;
                }
            } else {
                partition.split_off(a_r);
                if a_b != 0 && a_w != 0 { partition.split_off(a_b); }
                cid += 1;
            }
        } else {
            partition.split_off(a_r);
            if a_b != 0 && a_w != 0 { partition.split_off(a_b); }
            cid += 1;
        }
    }
    y_decrements
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn approx_2_lb(tf: &TwinForest) -> i32 {
    let n = tf.num_leaves;
    if n <= 1 { return 0; }

    let t1_info = build_tree_info(tf, T1);
    let t2_info = build_tree_info(tf, T2);

    let mut partition = Partition::new_single(n);
    let mut y_decrements: usize = 0;

    let mut is_roi_arr = vec![false; tf.num_nodes[T1]];
    let mut cover_owner = vec![0u32; tf.num_nodes[T1]];
    let mut cover_gen = vec![0u32; tf.num_nodes[T1]];
    let mut gen_ctr: u32 = 0;

    let max_iterations = 4 * n as usize;
    for _iter in 0..max_iterations {
        let u = match find_lowest_roi(
            tf, &partition, &t1_info.depth, &t2_info.depth,
            &t1_info.post_order, &t1_info.leaf_masks, &t2_info.leaf_masks,
            &mut is_roi_arr, &mut cover_owner, &mut cover_gen, &mut gen_ctr,
        ) {
            Some(u) => u,
            None => break,
        };

        y_decrements += 1;

        let lc = tf.left[T1][u as usize];
        let rc = tf.right[T1][u as usize];
        if lc == NONE || rc == NONE { continue; }

        let red = t1_info.leaf_masks[rc as usize];
        let blue = t1_info.leaf_masks[lc as usize];
        let all: u128 = (1u128 << n) - 1;
        let white = all & !(red | blue);

        y_decrements += make_rub_compatible(tf, &mut partition, &t2_info.depth, &t2_info.leaf_masks, red, blue);
        y_decrements += make_splittable(tf, &mut partition, &t2_info.depth, &t2_info.leaf_masks, red, blue, white);
        y_decrements += split_procedure(tf, &mut partition, &t2_info.depth, &t2_info.leaf_masks, red, blue, white);
    }

    let p = partition.count_components();
    let d = (p as i32) - 1 - (y_decrements as i32);
    d.max(0)
}
