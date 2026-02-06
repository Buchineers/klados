use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use klados_core::Tree;
use pace26io::binary_tree::{IndexedBinTree, IndexedBinTreeBuilder};
use pace26io::newick::BinaryTreeParser;
use std::fs;

fn load_instance(path: &str) -> Option<(Tree, IndexedBinTree, u32)> {
    if !std::path::Path::new(path).exists() {
        return None;
    }
    
    let content = fs::read_to_string(path).ok()?;
    let mut lines = content.lines();
    
    let header_line = lines
        .find(|l| l.starts_with("#p"))?;
    
    let parts: Vec<_> = header_line.split_whitespace().collect();
    let num_leaves = parts[2].parse::<u32>().ok()?;
    
    let tree_line = lines.find(|l| !l.trim().is_empty() && !l.starts_with('#'))?;
    
    let pace_tree = IndexedBinTreeBuilder::default()
        .parse_newick_from_str(tree_line.trim(), Default::default())
        .ok()?;
    
    let arena_tree = Tree::from_cursor(pace_tree.top_down(), num_leaves);
    
    Some((arena_tree, pace_tree, num_leaves))
}

fn count_leaves_pointer(tree: &IndexedBinTree) -> u32 {
    fn recurse(node: &IndexedBinTree, sum: &mut u32) {
        match node {
            pace26io::binary_tree::IndexedBinTree::Leaf(label) => {
                *sum = sum.wrapping_add(label.0);
            }
            pace26io::binary_tree::IndexedBinTree::Node(boxed) => {
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

fn bench_clone_scaling(c: &mut Criterion) {
    // Test different sizes with correct paths
    let instances = vec![
        ("10_leaves", "../../../stride-downloads/00/00/926dfbc847ee6009bf51ada7bc2b"),
        ("57_leaves", "../../../stride-downloads/02/e2/943b0192cc7cca00badd6265b93d"),
        ("100_leaves", "../../../stride-downloads/03/02/e3b8c284b4b99c20c7136c6edc0e"),
        ("192_leaves", "../../../stride-downloads/04/36/a0f3cbe3085c7ae4dc6ed05a74e9"),
    ];
    
    let mut group = c.benchmark_group("clone_scaling");
    
    for (name, path) in instances {
        if let Some((arena_tree, pointer_tree, num_leaves)) = load_instance(path) {
            let num_nodes = arena_tree.num_nodes();
            println!("\nLoaded {}: {} leaves, {} nodes", name, num_leaves, num_nodes);
            
            group.bench_with_input(
                BenchmarkId::new("arena", name),
                &num_nodes,
                |b, _| {
                    b.iter(|| {
                        let cloned = arena_tree.clone();
                        black_box(cloned.num_nodes())
                    });
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new("pointer", name),
                &num_nodes,
                |b, _| {
                    b.iter(|| {
                        let cloned = pointer_tree.clone();
                        black_box(count_leaves_pointer(&cloned))
                    });
                },
            );
        } else {
            println!("Skipping {} (not found)", name);
        }
    }
    
    group.finish();
}

criterion_group!(benches, bench_clone_scaling);
criterion_main!(benches);
