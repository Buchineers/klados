use klados_core::{Instance, Tree};
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;
use std::io::{self, BufReader};

#[path = "../commands/heuristic.rs"]
mod heuristic;

fn read_instance() -> Result<Instance, Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut builder = IndexedBinTreeBuilder::default();
    let pace = PaceInstance::try_read(reader, &mut builder)?;

    let num_leaves = pace.num_leaves as u32;
    let trees: Vec<Tree> = pace
        .trees
        .iter()
        .map(|t| Tree::from_cursor(t.top_down(), num_leaves))
        .collect();

    Ok(Instance::new(trees, num_leaves))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = read_instance()?;
    heuristic::run(&instance, "auto", false)
}
