//! Olver et al. 2-approximation dual lower bound for MAF on TwinForest.
//!
//! Direct ROI detection (checks all components at each T1 node) with O(1) LCA,
//! correct SPECIAL-SPLIT using T2-boundary topology.
//!
//! Returns D = |P| - 1 - y_decrements, a valid lower bound on OPT.

use super::forest::{TwinForest, T1, T2};
use klados_core::tree::{Label, NodeId, NONE};

const NO_COMP: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Sparse-table LCA (O(n) build, O(1) query)
// ---------------------------------------------------------------------------

struct LcaTable {
    first: Vec<u32>,
    depth: Vec<u32>,
    nodes: Vec<NodeId>,
    sparse: Vec<Vec<u32>>,
    log2: Vec<u32>,
}

impl LcaTable {
    fn build(tf: &TwinForest, ti: usize) -> Self {
        let n = tf.num_nodes[ti];
        let mut first = vec![u32::MAX; n];
        let cap = 2 * n + 1;
        let mut tour_depth = Vec::with_capacity(cap);
        let mut tour_nodes: Vec<NodeId> = Vec::with_capacity(cap);

        let mut stack: Vec<(NodeId, u32, bool)> = Vec::with_capacity(n);
        for &root in tf.components[ti].iter().rev() {
            stack.push((root, 0, false));
        }
        while let Some((node, d, is_return)) = stack.pop() {
            if node == NONE { continue; }
            let idx = tour_depth.len() as u32;
            tour_depth.push(d);
            tour_nodes.push(node);
            if !is_return && first[node as usize] == u32::MAX {
                first[node as usize] = idx;
            }
            if is_return { continue; }
            let lc = tf.left[ti][node as usize];
            let rc = tf.right[ti][node as usize];
            if lc == NONE && rc == NONE {
            } else if lc != NONE && rc != NONE {
                stack.push((node, d, true));
                stack.push((rc, d + 1, false));
                stack.push((node, d, true));
                stack.push((lc, d + 1, false));
            } else {
                let child = if lc != NONE { lc } else { rc };
                stack.push((node, d, true));
                stack.push((child, d + 1, false));
            }
        }

        let m = tour_depth.len();
        let mut log2 = vec![0u32; m + 2];
        for i in 2..=m { log2[i] = log2[i / 2] + 1; }
        let log_m = if m <= 1 { 1 } else { log2[m] as usize + 1 };
        let mut sparse = vec![vec![0u32; m]; log_m + 1];
        for i in 0..m { sparse[0][i] = i as u32; }
        let mut k = 1;
        while (1usize << k) <= m {
            for i in 0..=(m - (1 << k)) {
                let li = sparse[k - 1][i];
                let ri = sparse[k - 1][i + (1 << (k - 1))];
                sparse[k][i] = if tour_depth[li as usize] <= tour_depth[ri as usize] { li } else { ri };
            }
            k += 1;
        }
        LcaTable { first, depth: tour_depth, nodes: tour_nodes, sparse, log2 }
    }

    #[inline]
    fn lca(&self, a: NodeId, b: NodeId) -> NodeId {
        if a == NONE || b == NONE { return NONE; }
        let fa = self.first[a as usize];
        let fb = self.first[b as usize];
        if fa == u32::MAX || fb == u32::MAX { return NONE; }
        let (lo, hi) = if fa <= fb { (fa, fb) } else { (fb, fa) };
        let len = (hi - lo + 1) as usize;
        let k = self.log2[len] as usize;
        let li = self.sparse[k][lo as usize];
        let ri = self.sparse[k][hi as usize + 1 - (1 << k)];
        if self.depth[li as usize] <= self.depth[ri as usize] {
            self.nodes[li as usize]
        } else {
            self.nodes[ri as usize]
        }
    }

    /// Compute LCA of all labels in a mask, using label_to_node mapping.
    fn lca_of_mask(&self, tf: &TwinForest, ti: usize, mask: u128) -> NodeId {
        let mut result = NONE;
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros() as u32;
            m &= m - 1;
            let lbl = bit + 1;
            let node = tf.label_to_node[ti][lbl as usize];
            if result == NONE { result = node; }
            else { result = self.lca(result, node); }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Partition (bitmask-based for speed)
// ---------------------------------------------------------------------------

struct Partition {
    /// label -> component id
    comp: Vec<u32>,
    /// component id -> bitmask of labels (bit i = label i+1)
    masks: Vec<u128>,
    next_id: u32,
    n: u32,
}

impl Partition {
    fn new_single(n: u32) -> Self {
        let mut comp = vec![0u32; n as usize + 1];
        let mut mask: u128 = 0;
        for l in 1..=n { comp[l as usize] = 0; mask |= 1u128 << (l - 1); }
        Self { comp, masks: vec![mask], next_id: 1, n }
    }

    #[inline] fn component_of(&self, label: Label) -> u32 { self.comp[label as usize] }
    #[inline] fn comp_mask(&self, cid: u32) -> u128 { self.masks[cid as usize] }
    #[inline] fn comp_size(&self, cid: u32) -> u32 { self.masks[cid as usize].count_ones() }

    fn count_components(&self) -> usize { self.masks.iter().filter(|&&m| m != 0).count() }

    /// Split labels_mask off from their current component into a new component.
    fn split_off(&mut self, labels_mask: u128) {
        if labels_mask == 0 { return; }
        let first_lbl = labels_mask.trailing_zeros() + 1;
        let old_id = self.comp[first_lbl as usize];
        let new_id = self.next_id;
        self.next_id += 1;
        if new_id as usize >= self.masks.len() {
            self.masks.resize(new_id as usize + 1, 0);
        }
        // Update comp assignments
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
// Helpers
// ---------------------------------------------------------------------------

/// Precompute leaf_mask for each T1 node: bitmask of leaves under the node.
fn precompute_leaf_masks(tf: &TwinForest, ti: usize, post_order: &[NodeId]) -> Vec<u128> {
    let mut masks = vec![0u128; tf.num_nodes[ti]];
    for &node in post_order {
        if tf.is_leaf(ti, node) {
            let lbl = tf.label[ti][node as usize];
            if lbl != 0 { masks[node as usize] = 1u128 << (lbl - 1); }
        } else {
            let lc = tf.left[ti][node as usize];
            let rc = tf.right[ti][node as usize];
            if lc != NONE { masks[node as usize] |= masks[lc as usize]; }
            if rc != NONE { masks[node as usize] |= masks[rc as usize]; }
        }
    }
    masks
}

fn build_post_order(tf: &TwinForest, ti: usize) -> Vec<NodeId> {
    let mut order = Vec::with_capacity(tf.num_nodes[ti]);
    for &root in &tf.components[ti] {
        let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];
        while let Some((node, visited)) = stack.pop() {
            if node == NONE { continue; }
            if visited { order.push(node); }
            else {
                stack.push((node, true));
                let rc = tf.right[ti][node as usize];
                let lc = tf.left[ti][node as usize];
                if rc != NONE { stack.push((rc, false)); }
                if lc != NONE { stack.push((lc, false)); }
            }
        }
    }
    order
}



// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Find-Lowest-ROI: EXACT match to red_blue.rs logic
// ---------------------------------------------------------------------------

fn check_is_roi(
    tf: &TwinForest,
    partition: &Partition,
    lca1: &LcaTable,
    lca2: &LcaTable,
    u_leaves: u128,
    cover: &mut [u32],
    u: NodeId,
    t1_leaf_masks: &[u128],
    t2_leaf_masks: &[u128]
) -> bool {
    let num_comps = partition.masks.len() as u32;

    let u_L = tf.left[T1][u as usize];
    let u_R = tf.right[T1][u as usize];
    let u_L_leaves = if u_L != NONE { t1_leaf_masks[u_L as usize] } else { 0 };
    let u_R_leaves = if u_R != NONE { t1_leaf_masks[u_R as usize] } else { 0 };

    for cid in 0..num_comps {
        let a_mask = partition.masks[cid as usize];
        if a_mask == 0 { continue; }
        let inside = a_mask & u_leaves;
        if inside == 0 { continue; }

        let in_L = a_mask & u_L_leaves;
        let in_R = a_mask & u_R_leaves;
        if in_L != 0 && in_R != 0 {
            let lca_L = lca2.lca_of_mask(tf, T2, in_L);
            let lca_R = lca2.lca_of_mask(tf, T2, in_R);
            if lca_L != NONE && lca_R != NONE {
                // Pure ancestry check — no depth constraint.
                // If lca(L-labels in T2) has R-labels under it (or vice versa),
                // then L and R are interleaved in T2 → incompatible triple exists.
                let overlap = (t2_leaf_masks[lca_L as usize] & in_R) != 0
                           || (t2_leaf_masks[lca_R as usize] & in_L) != 0;
                if overlap {
                    return true;
                }
            }
        }

        // Condition C: component has labels both inside and outside L(u),
        // and adding some outside label to the inside set creates incompatibility.
        // Conservative check: entire component under lca_T2(inside).
        if inside != a_mask {
            let u_hat = lca2.lca_of_mask(tf, T2, inside);
            if u_hat != NONE {
                if (a_mask & t2_leaf_masks[u_hat as usize]) == a_mask {
                    return true;
                }
            }
        }
    }

    // Condition B (Overlap in T1)
    for i in 0..cover.len() { cover[i] = u32::MAX; }
    for cid in 0..num_comps {
        let mask = partition.masks[cid as usize];
        if mask == 0 { continue; }
        let inside = mask & u_leaves;
        if inside.count_ones() < 2 { continue; }
        let lca_node = lca1.lca_of_mask(tf, T1, inside);
        if lca_node == NONE { continue; }

        let mut m = inside;
        let mut conflict = false;
        while m != 0 {
            let bit = m.trailing_zeros();
            m &= m - 1;
            let mut cur = tf.label_to_node[T1][(bit as usize) + 1];
            while cur != NONE {
                if cover[cur as usize] == cid { break; }
                if cover[cur as usize] != u32::MAX && cover[cur as usize] != cid {
                    conflict = true; break;
                }
                cover[cur as usize] = cid;
                if cur == lca_node { break; }
                cur = tf.parent[T1][cur as usize];
            }
            if conflict { return true; }
        }
    }

    false
}

fn find_lowest_roi(
    tf: &TwinForest,
    partition: &Partition,
    lca1: &LcaTable,
    lca2: &LcaTable,
    t1_post_order: &[NodeId],
    t1_leaf_masks: &[u128],
    t2_leaf_masks: &[u128],
    is_roi_arr: &mut [bool],
    cover: &mut [u32],
) -> Option<NodeId> {
    is_roi_arr.fill(false);

    for &node in t1_post_order {
        if tf.is_leaf(T1, node) { continue; }
        is_roi_arr[node as usize] = check_is_roi(
            tf, partition, lca1, lca2,
            t1_leaf_masks[node as usize], cover,
            node, t1_leaf_masks, t2_leaf_masks
        );
    }

    // Find the lowest ROI
    for &node in t1_post_order {
        if tf.is_leaf(T1, node) { continue; }
        if is_roi_arr[node as usize] {
            let lc = tf.left[T1][node as usize];
            let rc = tf.right[T1][node as usize];
            let left_roi = lc != NONE && !tf.is_leaf(T1, lc) && is_roi_arr[lc as usize];
            let right_roi = rc != NONE && !tf.is_leaf(T1, rc) && is_roi_arr[rc as usize];
            if !left_roi && !right_roi {
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
    lca2: &LcaTable,
    t2_leaf_masks: &[u128],
    red: u128,
    blue: u128,
) -> usize {
    let mut num_splits = 0usize;
    loop {
        let mut found = false;
        for cid in 0..partition.masks.len() as u32 {
            let a_mask = partition.masks[cid as usize];
            if a_mask == 0 { continue; }
            let rub = a_mask & (red | blue);
            if rub.count_ones() < 2 { continue; }
            
            // O(1) Condition A check for R union B
            let in_R = a_mask & red;
            let in_B = a_mask & blue;
            if in_R != 0 && in_B != 0 {
                let lca_R = lca2.lca_of_mask(tf, T2, in_R);
                let lca_B = lca2.lca_of_mask(tf, T2, in_B);
                // Pure ancestry check — no depth constraint (same as check_is_roi).
                let overlap = lca_R != NONE && lca_B != NONE &&
                    ((t2_leaf_masks[lca_R as usize] & in_B) != 0 ||
                     (t2_leaf_masks[lca_B as usize] & in_R) != 0);
                if overlap {
                    // incompatible, we process this
                } else {
                    continue;
                }
            } else {
                continue;
            }

            // Incompatible R∪B subset — find T2 split point
            let mut u_hat = lca2.lca_of_mask(tf, T2, rub);
            if u_hat == NONE { continue; }
            
            loop {
                let lc = tf.left[T2][u_hat as usize];
                let rc = tf.right[T2][u_hat as usize];
                let mut moved = false;
                
                if lc != NONE {
                    let lc_leaves = t2_leaf_masks[lc as usize] & a_mask;
                    if (lc_leaves & red) != 0 && (lc_leaves & blue) != 0 {
                        u_hat = lc;
                        moved = true;
                    }
                }
                
                if !moved && rc != NONE {
                    let rc_leaves = t2_leaf_masks[rc as usize] & a_mask;
                    if (rc_leaves & red) != 0 && (rc_leaves & blue) != 0 {
                        u_hat = rc;
                        moved = true;
                    }
                }
                
                if !moved {
                    break;
                }
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
// Make-Splittable
// ---------------------------------------------------------------------------

fn make_splittable(
    tf: &TwinForest,
    partition: &mut Partition,
    lca2: &LcaTable,
    t2_post_order: &[NodeId],
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

            let lca_r = lca2.lca_of_mask(tf, T2, a_r);
            let lca_b = lca2.lca_of_mask(tf, T2, a_b);
            let lca_w = lca2.lca_of_mask(tf, T2, a_w);

            let overlap_rb = a_r.count_ones() >= 2 && a_b.count_ones() >= 2 && lca_r != NONE && lca_b != NONE && 
                (((t2_leaf_masks[lca_r as usize] & a_b) != 0 && tf.t2_depth[lca_r as usize] >= tf.t2_depth[lca_b as usize]) || 
                 ((t2_leaf_masks[lca_b as usize] & a_r) != 0 && tf.t2_depth[lca_b as usize] >= tf.t2_depth[lca_r as usize]));

            let overlap_rw = a_r.count_ones() >= 2 && a_w.count_ones() >= 2 && lca_r != NONE && lca_w != NONE && 
                (((t2_leaf_masks[lca_r as usize] & a_w) != 0 && tf.t2_depth[lca_r as usize] >= tf.t2_depth[lca_w as usize]) || 
                 ((t2_leaf_masks[lca_w as usize] & a_r) != 0 && tf.t2_depth[lca_w as usize] >= tf.t2_depth[lca_r as usize]));

            let overlap_bw = a_b.count_ones() >= 2 && a_w.count_ones() >= 2 && lca_b != NONE && lca_w != NONE && 
                (((t2_leaf_masks[lca_b as usize] & a_w) != 0 && tf.t2_depth[lca_b as usize] >= tf.t2_depth[lca_w as usize]) || 
                 ((t2_leaf_masks[lca_w as usize] & a_b) != 0 && tf.t2_depth[lca_w as usize] >= tf.t2_depth[lca_b as usize]));

            // If there's no overlap, it's already splittable.
            if !overlap_rb && !overlap_rw && !overlap_bw {
                continue;
            }

            // Find the deepest T2 node in the V-set intersection.
            // V(S) for label set S = {internal T2 nodes on paths from S-leaves to lca(S)}.
            // Node v ∈ V(S) iff: v is internal, descendant-or-equal of lca(S),
            // and t2_leaf_masks[v] & S != 0.
            let overlapping_pairs: [(u128, u128, NodeId, NodeId); 3] = [
                (a_r, a_b, lca_r, lca_b),
                (a_r, a_w, lca_r, lca_w),
                (a_b, a_w, lca_b, lca_w),
            ];
            let overlaps = [overlap_rb, overlap_rw, overlap_bw];

            let mut split_node = NONE;
            let mut best_depth: u16 = 0;

            for (idx, &(s1, s2, lca_s1, lca_s2)) in overlapping_pairs.iter().enumerate() {
                if !overlaps[idx] { continue; }
                // Find deepest node in V(s1) ∩ V(s2)
                for &node in t2_post_order {
                    if tf.is_leaf(T2, node) { continue; }
                    let leaves = t2_leaf_masks[node as usize];
                    // Must have leaves from both sets under it
                    if (leaves & s1) == 0 || (leaves & s2) == 0 { continue; }
                    // Must be descendant-or-equal of both LCAs
                    if lca2.lca(node, lca_s1) != lca_s1 { continue; }
                    if lca2.lca(node, lca_s2) != lca_s2 { continue; }
                    let d = tf.t2_depth[node as usize];
                    if split_node == NONE || d > best_depth {
                        split_node = node;
                        best_depth = d;
                    }
                }
                if split_node != NONE { break; }
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
// Split procedure (with correct SPECIAL-SPLIT using T2 topology)
// ---------------------------------------------------------------------------

fn split_procedure(
    tf: &TwinForest,
    partition: &mut Partition,
    lca2: &LcaTable,
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
            // Bicolored — standard split by color
            if a_r != 0 && (a_b != 0 || a_w != 0) {
                partition.split_off(a_r);
            } else if a_b != 0 && a_w != 0 {
                partition.split_off(a_b);
            }
            cid += 1;
            continue;
        }

        let rub_lca = lca2.lca_of_mask(tf, T2, a_r | a_b);
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
                    y_decrements += 1; // Critical for the Dual Bound!
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

    let lca1 = LcaTable::build(tf, T1);
    let lca2 = LcaTable::build(tf, T2);
    let t1_post_order = build_post_order(tf, T1);
    let t1_leaf_masks = precompute_leaf_masks(tf, T1, &t1_post_order);
    let t2_post_order = build_post_order(tf, T2);
    let t2_leaf_masks = precompute_leaf_masks(tf, T2, &t2_post_order);

    let mut partition = Partition::new_single(n);
    let mut y_decrements: usize = 0;

    let mut is_roi_arr = vec![false; tf.num_nodes[T1]];
    let mut cover = vec![u32::MAX; tf.num_nodes[T1]];

    let max_iterations = 4 * n as usize;
    for _iter in 0..max_iterations {
        let u = match find_lowest_roi(
            tf, &partition, &lca1, &lca2, &t1_post_order, &t1_leaf_masks, &t2_leaf_masks,
            &mut is_roi_arr, &mut cover
        ) {
            Some(u) => u,
            None => break,
        };

        y_decrements += 1;

        let lc = tf.left[T1][u as usize];
        let rc = tf.right[T1][u as usize];
        if lc == NONE || rc == NONE { continue; }

        let red = t1_leaf_masks[rc as usize];
        let blue = t1_leaf_masks[lc as usize];
        let all: u128 = (1u128 << n) - 1;
        let white = all & !(red | blue);

        y_decrements += make_rub_compatible(tf, &mut partition, &lca2, &t2_leaf_masks, red, blue);
        y_decrements += make_splittable(tf, &mut partition, &lca2, &t2_post_order, &t2_leaf_masks, red, blue, white);
        y_decrements += split_procedure(tf, &mut partition, &lca2, &t2_leaf_masks, red, blue, white);
    }

    let p = partition.count_components();
    let d = (p as i32) - 1 - (y_decrements as i32);
    d.max(0)
}
