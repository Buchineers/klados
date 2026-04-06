//! Delete a specific leaf from an instance and output the reduced instance.

use klados_core::Instance;
use klados_exact::kernelize;
use pace26io::newick::NewickWriter;
use std::io::{self, Write};

pub fn run(instance: &Instance, leaf: u32) -> Result<(), Box<dyn std::error::Error>> {
    if leaf < 1 || leaf > instance.num_leaves {
        eprintln!(
            "ERROR: leaf {} out of range [1, {}]",
            leaf, instance.num_leaves
        );
        return Ok(());
    }

    // Use reduce_instance with the leaf as "removed" from a dummy representative
    // Actually, reduce_instance collapses leaves. We want to DELETE.
    // Use a collapse where the "keep" is any other leaf and "remove" is our target,
    // but that changes semantics. Let me just build a FixedBitSet directly.
    //
    // Simpler: use the same approach as the pipeline — build keep set and restrict.
    let n = instance.num_leaves as usize;
    let mut keep_labels: Vec<u32> = (1..=instance.num_leaves).filter(|&l| l != leaf).collect();

    // Build label map
    let new_n = keep_labels.len() as u32;
    let mut label_map = vec![0u32; n + 1];
    let mut reverse_map = vec![0u32; new_n as usize + 1];
    for (new_idx, &old_lbl) in keep_labels.iter().enumerate() {
        let new_lbl = (new_idx + 1) as u32;
        label_map[old_lbl as usize] = new_lbl;
        reverse_map[new_lbl as usize] = old_lbl;
    }

    let trees: Vec<klados_core::Tree> = instance
        .trees
        .iter()
        .map(|t| {
            // Prune and relabel
            let mut keep_set = fixedbitset::FixedBitSet::with_capacity(n + 1);
            for &l in &keep_labels {
                keep_set.insert(l as usize);
            }
            let pruned = t.prune_to_leafset(&keep_set);
            pruned.relabel(&label_map, new_n)
        })
        .collect();

    let reduced = Instance::new(trees, new_n);

    let mut stdout = io::stdout().lock();
    writeln!(stdout, "#p {} {}", reduced.num_trees(), reduced.num_leaves)?;
    for tree in &reduced.trees {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }
    Ok(())
}
