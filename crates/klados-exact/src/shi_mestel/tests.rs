//! Unit tests for the Shi-Mestel solver.

use fixedbitset::FixedBitSet;
use klados_core::{Instance, Tree, XForest, NONE};

use super::branching::compute_e_f;
use super::forest_nav::component_leaf_sets_xf;
use super::reduction::{all_pairs_lsi_cached, find_all_sibling_pairs};
use super::search_state::SearchState;
use super::utils::{has_intersection, is_subset};
use super::ShiMestelSolver;

fn make_simple_tree() -> Tree {
    let mut tree = Tree::with_capacity(3);
    tree.parent.push(3);
    tree.left.push(NONE);
    tree.right.push(NONE);
    tree.label.push(1);
    tree.label_to_node[1] = 0;
    tree.parent.push(3);
    tree.left.push(NONE);
    tree.right.push(NONE);
    tree.label.push(2);
    tree.label_to_node[2] = 1;
    tree.parent.push(4);
    tree.left.push(NONE);
    tree.right.push(NONE);
    tree.label.push(3);
    tree.label_to_node[3] = 2;
    tree.parent.push(4);
    tree.left.push(0);
    tree.right.push(1);
    tree.label.push(0);
    tree.parent.push(NONE);
    tree.left.push(3);
    tree.right.push(2);
    tree.label.push(0);
    tree.root = 4;
    tree.compute_metadata();
    tree
}

#[test]
fn test_identical_trees_single_component() {
    let t1 = make_simple_tree();
    let t2 = make_simple_tree();
    let instance = Instance::new(vec![t1, t2], 3);
    let mut solver = ShiMestelSolver::new();
    let components = solver.solve(&instance).expect("solution");
    assert_eq!(components.len(), 1);
}

#[test]
fn test_xforest_from_tree() {
    let tree = make_simple_tree();
    let forest = XForest::from_tree(tree);
    assert_eq!(forest.cut_edges.count_ones(..), 0);
    assert_eq!(forest.component_roots.len(), 1);
}

#[test]
fn test_xforest_cut_uncut() {
    let tree = make_simple_tree();
    let mut forest = XForest::from_tree(tree);
    let orig_leafsets: Vec<FixedBitSet> = forest.live_leafsets.clone();
    forest.cut(0);
    assert!(forest.is_cut(0));
    assert_eq!(forest.cut_edges.count_ones(..), 1);
    forest.uncut(0);
    assert!(!forest.is_cut(0));
    assert_eq!(forest.cut_edges.count_ones(..), 0);
    assert_eq!(forest.live_leafsets, orig_leafsets);
}

#[test]
fn test_lsi_identical() {
    let t1 = make_simple_tree();
    let t2 = make_simple_tree();
    let f1 = XForest::from_tree(t1);
    let f2 = XForest::from_tree(t2);
    let forests = [f1, f2];
    let comp_sets: Vec<Vec<FixedBitSet>> = forests
        .iter()
        .map(|f| component_leaf_sets_xf(f, 3))
        .collect();
    assert!(all_pairs_lsi_cached(&comp_sets));
}

#[test]
fn test_component_label_sets() {
    let tree = make_simple_tree();
    let forest = XForest::from_tree(tree);
    let sets = component_leaf_sets_xf(&forest, 3);
    assert_eq!(sets.len(), 1);
}

#[test]
fn test_has_intersection() {
    let mut a = FixedBitSet::with_capacity(4);
    let mut b = FixedBitSet::with_capacity(4);
    a.insert(1);
    a.insert(2);
    b.insert(2);
    b.insert(3);
    assert!(has_intersection(&a, &b));
    b.clear();
    b.insert(3);
    assert!(!has_intersection(&a, &b));
}

#[test]
fn test_is_subset() {
    let mut a = FixedBitSet::with_capacity(4);
    let mut b = FixedBitSet::with_capacity(4);
    a.insert(1);
    a.insert(2);
    b.insert(1);
    b.insert(2);
    b.insert(3);
    assert!(is_subset(&a, &b));
    assert!(!is_subset(&b, &a));
}

#[test]
fn test_sibling_pair_detection() {
    let tree = make_simple_tree();
    let forest = XForest::from_tree(tree);
    let pairs = find_all_sibling_pairs(&forest, 3);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], (1, 2));
}

#[test]
fn test_e_f_siblings() {
    let tree = make_simple_tree();
    let forest = XForest::from_tree(tree);
    let e = compute_e_f(&forest, 1, 2);
    assert_eq!(e.len(), 0);
}

#[test]
fn test_search_state_checkpoint_rollback() {
    let t1 = make_simple_tree();
    let t2 = make_simple_tree();
    let mut state = SearchState::new(vec![XForest::from_tree(t1), XForest::from_tree(t2)]);
    let orig0: Vec<FixedBitSet> = state.forests[0].live_leafsets.clone();
    let orig1: Vec<FixedBitSet> = state.forests[1].live_leafsets.clone();

    state.checkpoint();
    state.cut_node(0, 0);
    assert!(state.forests[0].is_cut(0));
    state.rollback();
    assert!(!state.forests[0].is_cut(0));
    assert_eq!(state.forests[0].live_leafsets, orig0);
    assert_eq!(state.forests[1].live_leafsets, orig1);
}
