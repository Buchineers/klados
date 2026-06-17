//! Feasibility checking and root-of-infeasibility detection.

use crate::tree::{Label, NodeId};
use fixedbitset::FixedBitSet;
use log::debug;

use super::partition::Partition;
use super::tree_data::TreeData;

pub fn is_triple_compatible(
    td1: &TreeData,
    td2: &TreeData,
    x1: Label,
    x2: Label,
    x3: Label,
) -> bool {
    let n1 = td1.tree.label_to_node[x1 as usize];
    let n2 = td1.tree.label_to_node[x2 as usize];
    let n3 = td1.tree.label_to_node[x3 as usize];

    let lca12_1 = td1.lca(n1, n2);
    let lca123_1 = td1.lca(lca12_1, n3);
    let strict1 = lca12_1 != lca123_1;

    let m1 = td2.tree.label_to_node[x1 as usize];
    let m2 = td2.tree.label_to_node[x2 as usize];
    let m3 = td2.tree.label_to_node[x3 as usize];

    let lca12_2 = td2.lca(m1, m2);
    let lca123_2 = td2.lca(lca12_2, m3);
    let strict2 = lca12_2 != lca123_2;

    strict1 == strict2
}

pub fn is_set_compatible(td1: &TreeData, td2: &TreeData, labels: &FixedBitSet) -> bool {
    let lbls: Vec<usize> = labels.ones().collect();
    if lbls.len() <= 2 {
        return true;
    }

    for i in 0..lbls.len() {
        for j in 0..lbls.len() {
            if i == j {
                continue;
            }
            for k in 0..lbls.len() {
                if k == i || k == j {
                    continue;
                }
                if !is_triple_compatible(
                    td1,
                    td2,
                    lbls[i] as Label,
                    lbls[j] as Label,
                    lbls[k] as Label,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

pub fn mark_v_set(td: &TreeData, labels: &FixedBitSet, lca_node: NodeId) -> FixedBitSet {
    let num_nodes = td.tree.num_nodes();
    let mut marked = FixedBitSet::with_capacity(num_nodes);
    if lca_node == crate::NONE {
        return marked;
    }
    for lbl in labels.ones() {
        if lbl >= td.tree.label_to_node.len() {
            continue;
        }
        let mut cur = td.tree.label_to_node[lbl];
        if cur == crate::NONE {
            continue;
        }
        loop {
            if marked.contains(cur as usize) {
                break;
            }
            marked.insert(cur as usize);
            if cur == lca_node {
                break;
            }
            let p = td.tree.parent[cur as usize];
            if p == crate::NONE {
                break;
            }
            cur = p;
        }
    }
    if lca_node != crate::NONE {
        marked.insert(lca_node as usize);
    }
    marked
}

pub fn partition_overlaps_in_v2(td2: &TreeData, partition: &Partition) -> bool {
    let num_nodes = td2.tree.num_nodes();
    let mut cover: Vec<u32> = vec![u32::MAX; num_nodes];

    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        if labels.count_ones(..) < 2 {
            continue;
        }
        let lca_node = td2.lca_of_labels(labels);
        if lca_node == crate::NONE {
            continue;
        }
        for lbl in labels.ones() {
            if lbl >= td2.tree.label_to_node.len() {
                continue;
            }
            let mut cur = td2.tree.label_to_node[lbl];
            if cur == crate::NONE {
                continue;
            }
            loop {
                if cover[cur as usize] == cid {
                    break;
                }
                if cover[cur as usize] != u32::MAX && cover[cur as usize] != cid {
                    return true;
                }
                cover[cur as usize] = cid;
                if cur == lca_node {
                    break;
                }
                let p = td2.tree.parent[cur as usize];
                if p == crate::NONE {
                    break;
                }
                cur = p;
            }
        }
    }
    false
}

pub fn is_rub_feasible_impl(
    td1: &TreeData,
    td2: &TreeData,
    partition: &Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
) -> bool {
    let n = partition.n as usize;
    let rub = {
        let mut s = red.clone();
        s.union_with(blue);
        s
    };

    for w in 1..=n {
        for cid in partition.active_component_ids() {
            let comp = &partition.members[cid as usize];
            let mut test_set = comp.clone();
            let mut kw = rub.clone();
            kw.insert(w);
            test_set.intersect_with(&kw);
            if test_set.count_ones(..) >= 3 && !is_set_compatible(td1, td2, &test_set) {
                if log::log_enabled!(log::Level::Debug) {
                    let set_members: Vec<usize> = test_set.ones().collect();
                    let comp_members: Vec<usize> = comp.ones().collect();
                    debug!(
                        "[RB-FEA]     FAIL: comp {:?} with w={}: set {:?} not compatible",
                        comp_members, w, set_members
                    );
                }
                return false;
            }
        }
    }

    if partition_overlaps_in_v2(td2, partition) {
        debug!("[RB-FEA]     FAIL: partition overlaps in V2");
        return false;
    }

    let num_nodes = td1.tree.num_nodes();
    let mut cover: Vec<u32> = vec![u32::MAX; num_nodes];

    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        let mut inside = labels.clone();
        inside.intersect_with(&rub);

        if inside.count_ones(..) < 2 {
            continue;
        }

        let lca_node = td1.lca_of_labels(&inside);
        if lca_node == crate::NONE {
            continue;
        }

        for lbl in inside.ones() {
            let mut cur = td1.tree.label_to_node[lbl];
            if cur == crate::NONE {
                continue;
            }
            loop {
                if cover[cur as usize] == cid {
                    break;
                }
                if cover[cur as usize] != u32::MAX && cover[cur as usize] != cid {
                    debug!(
                        "[RB-FEA]     FAIL: overlap in V1[RUB] at node {} (comp {} and comp {})",
                        cur, cover[cur as usize], cid
                    );
                    return false;
                }
                cover[cur as usize] = cid;
                if cur == lca_node {
                    break;
                }
                let p = td1.tree.parent[cur as usize];
                if p == crate::NONE {
                    break;
                }
                cur = p;
            }
        }
    }

    true
}

pub fn is_roi(td1: &TreeData, td2: &TreeData, partition: &Partition, u: NodeId) -> bool {
    if td1.tree.is_leaf(u) {
        return false;
    }
    let u_leaves = &td1.leaf_set[u as usize];

    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        let mut intersection = labels.clone();
        intersection.intersect_with(u_leaves);
        let count = intersection.count_ones(..);
        if count >= 3 && !is_set_compatible(td1, td2, &intersection) {
            return true;
        }
    }

    {
        let num_nodes = td1.tree.num_nodes();
        let mut cover: Vec<u32> = vec![u32::MAX; num_nodes];
        for cid in partition.active_component_ids() {
            let labels = &partition.members[cid as usize];
            let mut inside = labels.clone();
            inside.intersect_with(u_leaves);
            if inside.count_ones(..) < 2 {
                continue;
            }
            let lca_node = td1.lca_of_labels(&inside);
            if lca_node == crate::NONE {
                continue;
            }
            for lbl in inside.ones() {
                if lbl >= td1.tree.label_to_node.len() {
                    continue;
                }
                let mut cur = td1.tree.label_to_node[lbl];
                if cur == crate::NONE {
                    continue;
                }
                loop {
                    if cover[cur as usize] == cid {
                        break;
                    }
                    if cover[cur as usize] != u32::MAX && cover[cur as usize] != cid {
                        return true;
                    }
                    cover[cur as usize] = cid;
                    if cur == lca_node {
                        break;
                    }
                    let p = td1.tree.parent[cur as usize];
                    if p == crate::NONE {
                        break;
                    }
                    cur = p;
                }
            }
        }
    }

    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        let mut inside = labels.clone();
        inside.intersect_with(u_leaves);
        if inside.count_ones(..) == 0 {
            continue;
        }
        let mut outside = labels.clone();
        for lbl in u_leaves.ones() {
            outside.set(lbl, false);
        }
        if outside.count_ones(..) == 0 {
            continue;
        }
        let all_incompatible = outside.ones().all(|w| {
            let mut test = inside.clone();
            test.insert(w);
            !is_set_compatible(td1, td2, &test)
        });
        if all_incompatible {
            return true;
        }
    }

    false
}

pub fn find_lowest_roi(td1: &TreeData, td2: &TreeData, partition: &Partition) -> Option<NodeId> {
    for &node in &td1.post_order {
        if td1.tree.is_leaf(node) {
            continue;
        }
        if is_roi(td1, td2, partition, node) {
            let (left, right) = td1.tree.children(node).unwrap();
            let left_roi = !td1.tree.is_leaf(left) && is_roi(td1, td2, partition, left);
            let right_roi = !td1.tree.is_leaf(right) && is_roi(td1, td2, partition, right);
            if !left_roi && !right_roi {
                return Some(node);
            }
        }
    }
    None
}
