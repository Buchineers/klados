use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use klados_core::Tree;
use pace26io::binary_tree::{IndexedBinTree, IndexedBinTreeBuilder};
use pace26io::newick::BinaryTreeParser;
use std::fs;

fn load_instance(path: &str) -> (Tree, IndexedBinTree, u32) {
    let content = fs::read_to_string(path).unwrap();
    let mut lines = content.lines();

    let header_line = lines.find(|l| l.starts_with("#p")).expect("No #p header");

    let parts: Vec<_> = header_line.split_whitespace().collect();
    let num_leaves = parts[2].parse::<u32>().unwrap();

    let tree_line = lines
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .expect("No tree found");

    let pace_tree = IndexedBinTreeBuilder::default()
        .parse_newick_from_str(tree_line.trim(), Default::default())
        .unwrap();

    let arena_tree = Tree::from_cursor(pace_tree.top_down(), num_leaves);

    (arena_tree, pace_tree, num_leaves)
}

#[inline(never)]
fn use_value<T>(val: T) -> T {
    val
}

fn setup_192() -> (Tree, IndexedBinTree, u32) {
    load_instance("../../../stride-downloads/04/36/a0f3cbe3085c7ae4dc6ed05a74e9")
}

// ==================== CLONE BENCHMARKS ====================

#[library_benchmark]
fn clone_arena_192() -> usize {
    let (tree, _, _) = setup_192();
    let cloned = tree.clone();
    use_value(cloned.num_nodes())
}

#[library_benchmark]
fn clone_pointer_192() -> u32 {
    let (_, tree, _) = setup_192();
    let cloned = tree.clone();
    count_leaves_pointer(&cloned)
}

// ==================== TRAVERSAL BENCHMARKS ====================

// Current: Iterator-based post-order (slower)
#[library_benchmark]
fn postorder_arena_iter_192() -> u32 {
    let (tree, _, _) = setup_192();
    let mut sum = 0u32;
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            if let Some(lbl) = tree.leaf_label(node) {
                sum = sum.wrapping_add(lbl);
            }
        }
    }
    use_value(sum)
}

// New: Recursive post-order (should be faster)
#[library_benchmark]
fn postorder_arena_recursive_192() -> u32 {
    let (tree, _, _) = setup_192();
    let sum = postorder_recursive(&tree, tree.root);
    use_value(sum)
}

fn postorder_recursive(tree: &Tree, node: u32) -> u32 {
    if tree.is_leaf(node) {
        tree.leaf_label(node).unwrap_or(0)
    } else {
        let (left, right) = tree.children(node).unwrap();
        postorder_recursive(tree, left).wrapping_add(postorder_recursive(tree, right))
    }
}

#[library_benchmark]
fn postorder_pointer_192() -> u32 {
    let (_, tree, _) = setup_192();
    count_leaves_pointer(&tree)
}

fn count_leaves_pointer(tree: &IndexedBinTree) -> u32 {
    fn recurse(node: &IndexedBinTree, sum: &mut u32) {
        match node {
            IndexedBinTree::Leaf(label) => {
                *sum = sum.wrapping_add(label.0);
            }
            IndexedBinTree::Node(boxed) => {
                let (_, left, right) = boxed.as_ref();
                recurse(left, sum);
                recurse(right, sum);
            }
        }
    }
    let mut sum = 0u32;
    recurse(tree, &mut sum);
    sum
}

// ==================== PARENT LOOKUP BENCHMARKS ====================

#[library_benchmark]
fn parent_lookups_192() -> u32 {
    let (tree, _, num_leaves) = setup_192();
    let mut total_depth = 0u32;
    for leaf in 1..=num_leaves {
        let mut node = tree.node_by_label(leaf);
        let mut depth = 0u32;
        while tree.parent[node as usize] != klados_core::NONE {
            node = tree.parent[node as usize];
            depth += 1;
        }
        total_depth = total_depth.wrapping_add(depth);
    }
    use_value(total_depth)
}

library_benchmark_group!(
    name = clone_benches;
    benchmarks = clone_arena_192, clone_pointer_192
);

library_benchmark_group!(
    name = traversal_benches;
    benchmarks = postorder_arena_iter_192, postorder_arena_recursive_192, postorder_pointer_192
);

library_benchmark_group!(
    name = access_benches;
    benchmarks = parent_lookups_192
);

main!(
    library_benchmark_groups = clone_benches,
    traversal_benches,
    access_benches
);
