//! Unit tests for lower bound computation.

use klados_core::tree::{Tree, NONE};

use super::tree_data::TreeData;
use super::{cherry_reduce_ub, maf_bounds, red_blue_approx};

fn make_tree_2leaves() -> (Tree, Tree) {
    let mut t1 = Tree::with_capacity(2);
    t1.parent.push(2);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(1);
    t1.label_to_node[1] = 0;
    t1.parent.push(2);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(2);
    t1.label_to_node[2] = 1;
    t1.parent.push(NONE);
    t1.left.push(0);
    t1.right.push(1);
    t1.label.push(0);
    t1.root = 2;
    t1.compute_metadata();
    (t1.clone(), t1)
}

fn make_3leaf_trees() -> (Tree, Tree) {
    let mut t1 = Tree::with_capacity(3);
    t1.parent.push(3);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(1);
    t1.label_to_node[1] = 0;
    t1.parent.push(3);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(2);
    t1.label_to_node[2] = 1;
    t1.parent.push(4);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(3);
    t1.label_to_node[3] = 2;
    t1.parent.push(4);
    t1.left.push(0);
    t1.right.push(1);
    t1.label.push(0);
    t1.parent.push(NONE);
    t1.left.push(3);
    t1.right.push(2);
    t1.label.push(0);
    t1.root = 4;
    t1.compute_metadata();

    let mut t2 = Tree::with_capacity(3);
    t2.parent.push(3);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(1);
    t2.label_to_node[1] = 0;
    t2.parent.push(3);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(3);
    t2.label_to_node[3] = 1;
    t2.parent.push(4);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(2);
    t2.label_to_node[2] = 2;
    t2.parent.push(4);
    t2.left.push(0);
    t2.right.push(1);
    t2.label.push(0);
    t2.parent.push(NONE);
    t2.left.push(3);
    t2.right.push(2);
    t2.label.push(0);
    t2.root = 4;
    t2.compute_metadata();
    (t1, t2)
}

fn make_4leaf_trees() -> (Tree, Tree) {
    let mut t1 = Tree::with_capacity(4);
    t1.parent.push(4);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(1);
    t1.label_to_node[1] = 0;
    t1.parent.push(4);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(2);
    t1.label_to_node[2] = 1;
    t1.parent.push(5);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(3);
    t1.label_to_node[3] = 2;
    t1.parent.push(5);
    t1.left.push(NONE);
    t1.right.push(NONE);
    t1.label.push(4);
    t1.label_to_node[4] = 3;
    t1.parent.push(6);
    t1.left.push(0);
    t1.right.push(1);
    t1.label.push(0);
    t1.parent.push(6);
    t1.left.push(2);
    t1.right.push(3);
    t1.label.push(0);
    t1.parent.push(NONE);
    t1.left.push(4);
    t1.right.push(5);
    t1.label.push(0);
    t1.root = 6;
    t1.compute_metadata();

    let mut t2 = Tree::with_capacity(4);
    t2.parent.push(4);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(1);
    t2.label_to_node[1] = 0;
    t2.parent.push(4);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(3);
    t2.label_to_node[3] = 1;
    t2.parent.push(5);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(2);
    t2.label_to_node[2] = 2;
    t2.parent.push(5);
    t2.left.push(NONE);
    t2.right.push(NONE);
    t2.label.push(4);
    t2.label_to_node[4] = 3;
    t2.parent.push(6);
    t2.left.push(0);
    t2.right.push(1);
    t2.label.push(0);
    t2.parent.push(6);
    t2.left.push(2);
    t2.right.push(3);
    t2.label.push(0);
    t2.parent.push(NONE);
    t2.left.push(4);
    t2.right.push(5);
    t2.label.push(0);
    t2.root = 6;
    t2.compute_metadata();
    (t1, t2)
}

#[test]
fn test_identical_trees() {
    let (t1, t2) = make_tree_2leaves();
    assert_eq!(cherry_reduce_ub(&t1, &t2), 0);
}

#[test]
fn test_lower_bound_identical() {
    let (t1, t2) = make_tree_2leaves();
    assert_eq!(maf_bounds(&[t1, t2], 2).lower, 1);
}

#[test]
fn test_red_blue_identical() {
    let (t1, t2) = make_tree_2leaves();
    assert_eq!(red_blue_approx(&t1, &t2), 0);
}

#[test]
fn test_red_blue_3leaf() {
    let (t1, t2) = make_3leaf_trees();
    let cost = red_blue_approx(&t1, &t2);
    assert!(cost >= 1 && cost <= 2, "red_blue cost={}", cost);
}

#[test]
fn test_red_blue_4leaf() {
    let (t1, t2) = make_4leaf_trees();
    let cost = red_blue_approx(&t1, &t2);
    assert!(cost >= 1 && cost <= 2, "red_blue cost={}", cost);
}

#[test]
fn test_lower_bound_3leaf() {
    let (t1, t2) = make_3leaf_trees();
    let lb = maf_bounds(&[t1, t2], 3).lower;
    assert!(lb >= 1 && lb <= 2, "lb={}", lb);
}

#[test]
fn test_single_tree() {
    let (t1, _) = make_3leaf_trees();
    assert_eq!(maf_bounds(&[t1], 3).lower, 1);
}

#[test]
fn test_lca_basic() {
    let (t1, _) = make_3leaf_trees();
    let td = TreeData::build(&t1);
    assert_eq!(td.lca(0, 1), 3);
    assert_eq!(td.lca(0, 2), 4);
    assert_eq!(td.lca(1, 2), 4);
}
