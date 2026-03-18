//! Red-Blue 2-approximation algorithm for MAF lower bounds.

use fixedbitset::FixedBitSet;
use crate::tree::{Label, NONE, NodeId, Tree};

use super::feasibility::{
    find_lowest_roi, is_rub_feasible_impl, is_set_compatible, is_triple_compatible, mark_v_set,
};
use super::partition::Partition;
use super::tree_data::TreeData;

/// Result of the detailed red-blue 2-approximation algorithm.
pub struct RedBlueResult {
    /// Upper bound: the solution cost (number of components - 1 after merging).
    pub ub: usize,
    /// Dual lower bound D on OPT: |P_before_merge| - 1 - y_decrements.
    pub dual_lb: usize,
}

pub fn red_blue_approx_detailed(t1: &Tree, t2: &Tree) -> RedBlueResult {
    let n = t1.num_leaves;
    if n <= 1 {
        return RedBlueResult { ub: 0, dual_lb: 0 };
    }

    let trace = std::env::var("RED_BLUE_TRACE").is_ok();

    let td1 = TreeData::build(t1);
    let td2 = TreeData::build(t2);
    let mut partition = Partition::new_single(n);
    let mut pairslist: Vec<(Label, Label)> = Vec::new();
    let mut y_decrements: usize = 0;

    if trace {
        eprintln!("[RB] Starting red_blue_approx_detailed with {} leaves", n);
        eprintln!("[RB] T1 root={}, T2 root={}", t1.root, t2.root);
    }

    let max_iterations = 4 * n as usize;
    for iter in 0..max_iterations {
        if trace {
            eprintln!("[RB] === Iteration {} ===", iter);
            eprintln!(
                "[RB] Partition has {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        let u = match find_lowest_roi(&td1, &td2, &partition) {
            Some(u) => u,
            None => {
                if trace {
                    eprintln!("[RB] No ROI found — partition is feasible");
                }
                break;
            }
        };

        // y decrement for lca1(R∪B) at start of iteration
        y_decrements += 1;

        let (u_l, u_r) = t1.children(u).unwrap();

        let red = &td1.leaf_set[u_r as usize];
        let blue = &td1.leaf_set[u_l as usize];
        let mut white = FixedBitSet::with_capacity(n as usize + 1);
        for lbl in 1..=n as usize {
            if !red.contains(lbl) && !blue.contains(lbl) {
                white.insert(lbl);
            }
        }

        if trace {
            let reds: Vec<usize> = red.ones().collect();
            let blues: Vec<usize> = blue.ones().collect();
            let whites: Vec<usize> = white.ones().collect();
            eprintln!("[RB] ROI: u={} (u_l={}, u_r={})", u, u_l, u_r);
            eprintln!("[RB] Red (L(u_r)): {:?}", reds);
            eprintln!("[RB] Blue (L(u_l)): {:?}", blues);
            eprintln!("[RB] White: {:?}", whites);
        }

        let original_comp = partition.comp.clone();

        y_decrements += make_rub_compatible(&td1, &td2, &mut partition, red, blue);
        if trace {
            eprintln!(
                "[RB] After Make-RUB-compatible: {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        y_decrements += make_splittable(&td2, &mut partition, red, blue, &white);
        if trace {
            eprintln!(
                "[RB] After Make-Splittable: {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        y_decrements += split_procedure(&td1, &td2, &mut partition, red, blue, &white);
        if trace {
            eprintln!(
                "[RB] After Split: {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        if let Some(pair) = find_merge_pair(&td1, &td2, &partition, red, blue, &original_comp) {
            if trace {
                eprintln!("[RB] Find-Merge-Pair found: ({}, {})", pair.0, pair.1);
            }
            pairslist.push(pair);
        } else if trace {
            eprintln!("[RB] Find-Merge-Pair: no pair found");
        }
    }

    // D = |P_before_merge| - 1 - y_decrements
    let p_before_merge = partition.count_components();
    let dual_lb = p_before_merge.saturating_sub(1).saturating_sub(y_decrements);

    if trace {
        eprintln!("[RB] === Merge-Components: {} pairs ===", pairslist.len());
        eprintln!(
            "[RB] Dual: |P_before_merge|={}, y_decrements={}, D={}",
            p_before_merge, y_decrements, dual_lb
        );
    }
    for &(x1, x2) in pairslist.iter() {
        let c1 = partition.component_of(x1);
        let c2 = partition.component_of(x2);
        if trace {
            eprintln!("[RB] Merge: x1={}, x2={}, c1={}, c2={}", x1, x2, c1, c2);
        }
        partition.merge(c1, c2);
    }

    let nc = partition.count_components();
    if trace {
        eprintln!(
            "[RB] Final: {} components (UB={}, dual_LB={})",
            nc,
            nc.saturating_sub(1),
            dual_lb,
        );
        for cid in partition.active_component_ids() {
            let members: Vec<usize> = partition.members[cid as usize].ones().collect();
            eprintln!("[RB]   comp {}: {:?}", cid, members);
        }
    }
    let ub = if nc == 0 { 0 } else { nc - 1 };
    RedBlueResult { ub, dual_lb }
}

pub fn red_blue_approx(t1: &Tree, t2: &Tree) -> usize {
    let n = t1.num_leaves;
    if n <= 1 {
        return 0;
    }

    let trace = std::env::var("RED_BLUE_TRACE").is_ok();

    let td1 = TreeData::build(t1);
    let td2 = TreeData::build(t2);
    let mut partition = Partition::new_single(n);
    let mut pairslist: Vec<(Label, Label)> = Vec::new();

    if trace {
        eprintln!("[RB] Starting red_blue_approx with {} leaves", n);
        eprintln!("[RB] T1 root={}, T2 root={}", t1.root, t2.root);
    }

    let max_iterations = 4 * n as usize;
    for iter in 0..max_iterations {
        if trace {
            eprintln!("[RB] === Iteration {} ===", iter);
            eprintln!(
                "[RB] Partition has {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        let u = match find_lowest_roi(&td1, &td2, &partition) {
            Some(u) => u,
            None => {
                if trace {
                    eprintln!("[RB] No ROI found — partition is feasible");
                }
                break;
            }
        };

        let (u_l, u_r) = t1.children(u).unwrap();

        let red = &td1.leaf_set[u_r as usize];
        let blue = &td1.leaf_set[u_l as usize];
        let mut white = FixedBitSet::with_capacity(n as usize + 1);
        for lbl in 1..=n as usize {
            if !red.contains(lbl) && !blue.contains(lbl) {
                white.insert(lbl);
            }
        }

        if trace {
            let reds: Vec<usize> = red.ones().collect();
            let blues: Vec<usize> = blue.ones().collect();
            let whites: Vec<usize> = white.ones().collect();
            eprintln!("[RB] ROI: u={} (u_l={}, u_r={})", u, u_l, u_r);
            eprintln!("[RB] Red (L(u_r)): {:?}", reds);
            eprintln!("[RB] Blue (L(u_l)): {:?}", blues);
            eprintln!("[RB] White: {:?}", whites);
        }

        let original_comp = partition.comp.clone();

        let _ = make_rub_compatible(&td1, &td2, &mut partition, red, blue);
        if trace {
            eprintln!(
                "[RB] After Make-RUB-compatible: {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        let _ = make_splittable(&td2, &mut partition, red, blue, &white);
        if trace {
            eprintln!(
                "[RB] After Make-Splittable: {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        let _ = split_procedure(&td1, &td2, &mut partition, red, blue, &white);
        if trace {
            eprintln!(
                "[RB] After Split: {} components",
                partition.count_components()
            );
            for cid in partition.active_component_ids() {
                let members: Vec<usize> = partition.members[cid as usize].ones().collect();
                eprintln!("[RB]   comp {}: {:?}", cid, members);
            }
        }

        if let Some(pair) = find_merge_pair(&td1, &td2, &partition, red, blue, &original_comp) {
            if trace {
                eprintln!("[RB] Find-Merge-Pair found: ({}, {})", pair.0, pair.1);
            }
            pairslist.push(pair);
        } else if trace {
            eprintln!("[RB] Find-Merge-Pair: no pair found");
        }
    }

    if trace {
        eprintln!("[RB] === Merge-Components: {} pairs ===", pairslist.len());
    }
    for &(x1, x2) in pairslist.iter() {
        let c1 = partition.component_of(x1);
        let c2 = partition.component_of(x2);
        if trace {
            eprintln!("[RB] Merge: x1={}, x2={}, c1={}, c2={}", x1, x2, c1, c2);
        }
        partition.merge(c1, c2);
    }

    let nc = partition.count_components();
    if trace {
        eprintln!(
            "[RB] Final: {} components (cost={})",
            nc,
            nc.saturating_sub(1)
        );
        for cid in partition.active_component_ids() {
            let members: Vec<usize> = partition.members[cid as usize].ones().collect();
            eprintln!("[RB]   comp {}: {:?}", cid, members);
        }
    }
    if nc == 0 { 0 } else { nc - 1 }
}

/// Returns the number of splits performed (= number of y decrements).
fn make_rub_compatible(
    td1: &TreeData,
    td2: &TreeData,
    partition: &mut Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
) -> usize {
    let mut rub = red.clone();
    rub.union_with(blue);
    let mut num_splits = 0usize;

    loop {
        let mut split_done = false;
        for cid in partition.active_component_ids() {
            let labels = &partition.members[cid as usize];
            let mut a_rub = labels.clone();
            a_rub.intersect_with(&rub);
            if a_rub.count_ones(..) < 2 {
                continue;
            }

            if is_set_compatible(td1, td2, &a_rub) {
                continue;
            }

            if let Some(u_hat) = find_rub_split_node(td2, labels, red, blue) {
                let u_hat_leaves = &td2.leaf_set[u_hat as usize];
                let mut inside = labels.clone();
                inside.intersect_with(u_hat_leaves);
                if inside.count_ones(..) > 0 && inside.count_ones(..) < labels.count_ones(..) {
                    partition.split_off(&inside);
                    num_splits += 1;
                    split_done = true;
                    break;
                }
            }
        }
        if !split_done {
            break;
        }
    }
    num_splits
}

fn find_rub_split_node(
    td2: &TreeData,
    comp_labels: &FixedBitSet,
    red: &FixedBitSet,
    blue: &FixedBitSet,
) -> Option<NodeId> {
    if comp_labels.count_ones(..) < 2 {
        return None;
    }
    let comp_lca = td2.lca_of_labels(comp_labels);
    if comp_lca == NONE {
        return None;
    }

    let mut best: Option<NodeId> = None;
    let mut best_depth: u16 = 0;

    let mut stack = vec![comp_lca];
    while let Some(v) = stack.pop() {
        if td2.tree.is_leaf(v) {
            continue;
        }
        let mut cv = td2.leaf_set[v as usize].clone();
        cv.intersect_with(comp_labels);

        let mut has_red = false;
        let mut has_blue = false;
        for lbl in cv.ones() {
            if red.contains(lbl) {
                has_red = true;
            }
            if blue.contains(lbl) {
                has_blue = true;
            }
            if has_red && has_blue {
                break;
            }
        }

        if has_red && has_blue {
            let d = td2.tree.depth[v as usize];
            if best.is_none() || d > best_depth {
                best = Some(v);
                best_depth = d;
            }
        }

        if let Some((left, right)) = td2.tree.children(v) {
            if !td2.leaf_set[left as usize].is_disjoint(comp_labels) {
                stack.push(left);
            }
            if !td2.leaf_set[right as usize].is_disjoint(comp_labels) {
                stack.push(right);
            }
        }
    }
    best
}

/// Returns the number of splits performed (= number of y decrements).
fn make_splittable(
    td2: &TreeData,
    partition: &mut Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) -> usize {
    let mut num_splits = 0usize;
    loop {
        let mut split_done = false;
        for cid in partition.active_component_ids() {
            let labels = &partition.members[cid as usize];
            if is_splittable(td2, labels, red, blue, white) {
                continue;
            }
            if let Some(u_hat) = find_splittable_split_node(td2, labels, red, blue, white) {
                let u_hat_leaves = &td2.leaf_set[u_hat as usize];
                let mut inside = labels.clone();
                inside.intersect_with(u_hat_leaves);
                if inside.count_ones(..) > 0 && inside.count_ones(..) < labels.count_ones(..) {
                    partition.split_off(&inside);
                    num_splits += 1;
                    split_done = true;
                    break;
                }
            }
        }
        if !split_done {
            break;
        }
    }
    num_splits
}

fn is_splittable(
    td2: &TreeData,
    labels: &FixedBitSet,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) -> bool {
    let mut a_r = labels.clone();
    a_r.intersect_with(red);
    let mut a_b = labels.clone();
    a_b.intersect_with(blue);
    let mut a_w = labels.clone();
    a_w.intersect_with(white);

    !sets_overlap_in_v2(td2, &a_r, &a_b)
        && !sets_overlap_in_v2(td2, &a_r, &a_w)
        && !sets_overlap_in_v2(td2, &a_b, &a_w)
}

fn sets_overlap_in_v2(td2: &TreeData, s1: &FixedBitSet, s2: &FixedBitSet) -> bool {
    if s1.count_ones(..) < 2 || s2.count_ones(..) < 2 {
        return false;
    }
    let lca1 = td2.lca_of_labels(s1);
    let lca2 = td2.lca_of_labels(s2);
    if lca1 == NONE || lca2 == NONE {
        return false;
    }

    let v1_set = mark_v_set(td2, s1, lca1);
    let v2_set = mark_v_set(td2, s2, lca2);
    !v1_set.is_disjoint(&v2_set)
}

fn find_splittable_split_node(
    td2: &TreeData,
    labels: &FixedBitSet,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) -> Option<NodeId> {
    let mut a_r = labels.clone();
    a_r.intersect_with(red);
    let mut a_b = labels.clone();
    a_b.intersect_with(blue);
    let mut a_w = labels.clone();
    a_w.intersect_with(white);

    let pairs: [(&FixedBitSet, &FixedBitSet); 3] = [(&a_r, &a_b), (&a_r, &a_w), (&a_b, &a_w)];
    for (s1, s2) in pairs {
        if s1.count_ones(..) >= 1 && s2.count_ones(..) >= 1 {
            let lca_s1 = td2.lca_of_labels(s1);
            let lca_s2 = td2.lca_of_labels(s2);
            if lca_s1 == NONE || lca_s2 == NONE {
                continue;
            }
            if s1.count_ones(..) >= 2 && s2.count_ones(..) >= 2 {
                let v1_set = mark_v_set(td2, s1, lca_s1);
                let v2_set = mark_v_set(td2, s2, lca_s2);
                if !v1_set.is_disjoint(&v2_set) {
                    let mut best: Option<NodeId> = None;
                    let mut best_depth = 0u16;
                    let intersection = {
                        let mut r = v1_set;
                        r.intersect_with(&v2_set);
                        r
                    };
                    for node_idx in intersection.ones() {
                        let d = td2.tree.depth[node_idx];
                        if best.is_none() || d > best_depth {
                            best = Some(node_idx as NodeId);
                            best_depth = d;
                        }
                    }
                    return best;
                }
            }
        }
    }
    None
}

/// Returns the number of y decrements from special_split's else branch.
fn split_procedure(
    td1: &TreeData,
    td2: &TreeData,
    partition: &mut Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) -> usize {
    let mut y_decrements = 0usize;
    let comp_ids = partition.active_component_ids();
    for cid in comp_ids {
        let labels = partition.members[cid as usize].clone();
        let mut a_r = labels.clone();
        a_r.intersect_with(red);
        let mut a_b = labels.clone();
        a_b.intersect_with(blue);
        let mut a_w = labels.clone();
        a_w.intersect_with(white);

        let num_colors = (a_r.count_ones(..) > 0) as u8
            + (a_b.count_ones(..) > 0) as u8
            + (a_w.count_ones(..) > 0) as u8;
        if num_colors <= 1 {
            continue;
        }

        let is_tricolored =
            a_r.count_ones(..) > 0 && a_b.count_ones(..) > 0 && a_w.count_ones(..) > 0;

        if is_tricolored && has_compatible_tricolored_triple(td1, td2, &a_r, &a_b, &a_w) {
            y_decrements += special_split(td1, td2, partition, cid, &a_r, &a_b, &a_w);
        } else {
            if a_r.count_ones(..) > 0 && (a_b.count_ones(..) > 0 || a_w.count_ones(..) > 0) {
                partition.split_off(&a_r);
            }
            if a_b.count_ones(..) > 0 && a_w.count_ones(..) > 0 {
                partition.split_off(&a_b);
            }
        }
    }
    y_decrements
}

fn has_compatible_tricolored_triple(
    td1: &TreeData,
    td2: &TreeData,
    a_r: &FixedBitSet,
    a_b: &FixedBitSet,
    a_w: &FixedBitSet,
) -> bool {
    for r in a_r.ones() {
        for b in a_b.ones() {
            for w in a_w.ones() {
                if is_triple_compatible(td1, td2, r as Label, b as Label, w as Label) {
                    return true;
                }
            }
        }
    }
    false
}

/// Returns 1 if the else branch was executed (y decrement for û = lca2(A∩(R∪B))), 0 otherwise.
fn special_split(
    td1: &TreeData,
    td2: &TreeData,
    partition: &mut Partition,
    _cid: u32,
    a_r: &FixedBitSet,
    a_b: &FixedBitSet,
    a_w: &FixedBitSet,
) -> usize {
    let all_compatible = a_r.ones().all(|r| {
        a_b.ones().all(|b| {
            a_w.ones()
                .all(|w| is_triple_compatible(td1, td2, r as Label, b as Label, w as Label))
        })
    });

    if all_compatible {
        partition.split_off(a_r);
        0
    } else {
        let mut rub = a_r.clone();
        rub.union_with(a_b);
        if rub.count_ones(..) == 0 {
            return 0;
        }
        let u_hat = td2.lca_of_labels(&rub);
        let u_hat_leaves = &td2.leaf_set[u_hat as usize];

        let mut all_labels = a_r.clone();
        all_labels.union_with(a_b);
        all_labels.union_with(a_w);

        let mut a_prime = all_labels.clone();
        a_prime.intersect_with(u_hat_leaves);

        let mut a_outside = all_labels.clone();
        for lbl in a_prime.ones() {
            a_outside.set(lbl, false);
        }

        if a_outside.count_ones(..) > 0 {
            partition.split_off(&a_outside);
        }

        let mut apr = a_prime.clone();
        apr.intersect_with(a_r);
        let mut apb = a_prime.clone();
        apb.intersect_with(a_b);

        if apr.count_ones(..) > 0 {
            partition.split_off(&apr);
        }
        if apb.count_ones(..) > 0 {
            partition.split_off(&apb);
        }
        1
    }
}

fn find_merge_pair(
    td1: &TreeData,
    td2: &TreeData,
    partition: &Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    original_comp: &[u32],
) -> Option<(Label, Label)> {
    let trace = std::env::var("RED_BLUE_TRACE").is_ok();
    let mut rub = red.clone();
    rub.union_with(blue);
    let rub_labels: Vec<usize> = rub.ones().collect();

    if trace {
        eprintln!(
            "[RB-FMP] Searching for merge pair among R∪B labels: {:?}",
            rub_labels
        );
    }

    for i in 0..rub_labels.len() {
        for j in (i + 1)..rub_labels.len() {
            let x1 = rub_labels[i] as Label;
            let x2 = rub_labels[j] as Label;

            if original_comp[x1 as usize] != original_comp[x2 as usize] {
                if trace {
                    eprintln!(
                        "[RB-FMP]   ({},{}) skipped: different original component",
                        x1, x2
                    );
                }
                continue;
            }

            let c1 = partition.component_of(x1);
            let c2 = partition.component_of(x2);
            if c1 == c2 {
                if trace {
                    eprintln!("[RB-FMP]   ({},{}) skipped: same component now", x1, x2);
                }
                continue;
            }

            if trace {
                let m1: Vec<usize> = partition.members[c1 as usize].ones().collect();
                let m2: Vec<usize> = partition.members[c2 as usize].ones().collect();
                eprintln!(
                    "[RB-FMP]   Checking ({},{}) : comps {:?} and {:?}",
                    x1, x2, m1, m2
                );
            }

            let mut test = Partition {
                comp: partition.comp.clone(),
                members: partition.members.clone(),
                n: partition.n,
            };
            test.merge(c1, c2);

            let feasible = is_rub_feasible_impl(td1, td2, &test, red, blue, trace);
            if feasible {
                if trace {
                    eprintln!(
                        "[RB-FMP]   ({},{}) is R∪B-feasible! Returning pair.",
                        x1, x2
                    );
                }
                return Some((x1, x2));
            } else if trace {
                eprintln!("[RB-FMP]   ({},{}) is NOT R∪B-feasible", x1, x2);
            }
        }
    }
    None
}
