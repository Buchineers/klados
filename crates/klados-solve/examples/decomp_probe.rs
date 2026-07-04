//! Decomposability-under-branching diagnostic.
//!
//! Tests the central bet: an irreducible core that does NOT decompose at the
//! root DOES decompose once the search has committed some decisions. Ryan-Foster
//! branching + the 3-2 kernelization rule realize their effect by fixing leaves
//! as singletons (cuts). We simulate that by progressively REMOVING cut leaves
//! from the core and, at each step, re-kernelizing + running `find_clusters` on
//! the residual. If the largest remaining subproblem shrinks steeply, interleaved
//! decomposition/kernelization will factorize the exact search → the win. If it
//! stays monolithic until nearly all cuts are made, the core is dense → we need a
//! stronger *static* decomposition instead.
//!
//! Two cut orders are reported:
//!   - `chen`  : singletons of the Chen 2-approx AF (a real, structure-aware cut
//!               set; over-cuts vs optimal, so treat the x-axis as an upper bound
//!               on how many *true* cuts are needed).
//!   - `random`: uniformly random leaves (a pessimistic baseline — the search does
//!               NOT cut randomly, so `chen` should decompose earlier than this).
//!
//! Usage:
//!   cargo run --release --example decomp_probe -- <file> [<file> ...]
//! Env:
//!   DECOMP_PROBE_STEPS   number of fractions between 0 and 1 (default 10)
//!   DECOMP_PROBE_SEED    RNG seed for the random baseline (default 1)

use fixedbitset::FixedBitSet;
use klados_core::Instance;
use klados_core::cluster_decomposition::find_clusters;
use klados_core::kernelize::{self, KernelizeConfig};
use klados_solve::solvers::bp::bp_solve_capped_until;
use klados_solve::solvers::chen_rspr::chen_pair_agreement;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

/// Largest exact subproblem the search would still face after one decomposition
/// pass on `inst`: max over (each cluster size, the remainder size).
fn largest_subproblem(inst: &Instance) -> (usize, usize, usize) {
    let n = inst.num_leaves as usize;
    match find_clusters(inst) {
        None => (0, n, n), // no split: the whole residual is one subproblem
        Some(clusters) => {
            let mut clustered = 0usize;
            let mut largest_cluster = 0usize;
            for c in &clusters {
                let s = c.leaves.count_ones(..);
                clustered += s;
                largest_cluster = largest_cluster.max(s);
            }
            let remainder = n - clustered + clusters.len();
            (clusters.len(), largest_cluster, largest_cluster.max(remainder))
        }
    }
}

/// Kernelize `inst` and return (kernelized_leaves, largest_subproblem_after_split).
fn probe_point(inst: &Instance) -> (usize, usize, usize, usize) {
    let kern = kernelize::kernelize_best(inst, &KernelizeConfig::default());
    let red = &kern.instance;
    let (nclusters, largest_cluster, largest_sub) = largest_subproblem(red);
    (red.num_leaves as usize, nclusters, largest_cluster, largest_sub)
}

/// Build a residual keeping all core labels EXCEPT the first `k` of `cut_order`.
fn residual_without(core: &Instance, cut_order: &[u32], k: usize) -> Instance {
    let n = core.num_leaves as usize;
    let mut keep = FixedBitSet::with_capacity(n + 1);
    keep.insert_range(1..(n + 1));
    for &lbl in cut_order.iter().take(k) {
        keep.set(lbl as usize, false);
    }
    kernelize::restrict_instance_simple(core, &keep).0
}

/// MUST-LINK contraction (the faithful B&P reducing op). Contract the first `k`
/// agreement components in `comps` — each is a clade in both trees, so keeping
/// one representative leaf and dropping the rest IS a topology-preserving
/// contraction to a super-leaf. All other leaves (singletons + not-yet-assembled
/// components) are kept. This models the search assembling components via
/// must-link and reducing each to a placeholder.
fn residual_contract(core: &Instance, comps: &[Vec<u32>], k: usize) -> Instance {
    let n = core.num_leaves as usize;
    let mut keep = FixedBitSet::with_capacity(n + 1);
    keep.insert_range(1..(n + 1));
    for comp in comps.iter().take(k) {
        // keep comp[0] as the representative super-leaf, drop the rest
        for &lbl in &comp[1..] {
            keep.set(lbl as usize, false);
        }
    }
    kernelize::restrict_instance_simple(core, &keep).0
}

fn run_one(path: &str, steps: usize, seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::from_file(Path::new(path))?;
    let n0 = instance.num_leaves;
    let m = instance.num_trees();

    // Kernelize the raw instance to the irreducible core.
    let kern = kernelize::kernelize_best(&instance, &KernelizeConfig::default());
    let core = kern.instance.clone();
    let core_n = core.num_leaves as usize;

    println!("\n=== {path} ===");
    println!("  raw n={n0} m={m}  ->  kernelized core n={core_n}");
    if m != 2 {
        println!("  (skip: probe is 2-tree only)");
        return Ok(());
    }
    if core_n < 8 {
        println!("  (skip: core too small)");
        return Ok(());
    }

    // Root decomposition (baseline).
    let (rk_n, rk_cl, rk_lc, rk_ls) = probe_point(&core);
    println!(
        "  ROOT   : kern={rk_n} clusters={rk_cl} largest_cluster={rk_lc} LARGEST_SUBPROBLEM={rk_ls} ({:.0}% of core)",
        100.0 * rk_ls as f64 / core_n as f64
    );

    // Chen 2-approx cut set = singleton components of the Chen AF (on the core).
    let (_lo, chen_up, chen_sets) = chen_pair_agreement(&core.trees[0], &core.trees[1]);
    let mut chen_cuts: Vec<u32> = chen_sets
        .iter()
        .filter(|c| c.len() == 1)
        .map(|c| c[0])
        .collect();
    println!(
        "  chen AF: {} comps (2-approx ub={chen_up}), {} singletons (cut set)",
        chen_sets.len(),
        chen_cuts.len()
    );

    let _ = (seed, chen_up);
    let _ = &chen_cuts;

    // Solve the core (or best within the deadline) to get the OPTIMAL AF. From it
    // we drive TWO faithful commit orders toward the optimum:
    //   CONTRACT (must-link): assemble multi-leaf components, each contracted to a
    //     super-leaf (keep 1 rep). This is the B&P reducing operation.
    //   CUT (cannot-link/3-2): remove singleton leaves. The rspr-style cut op.
    let exact_secs: u64 = std::env::var("DECOMP_PROBE_EXACT_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let terminate = Arc::new(AtomicBool::new(false));
    let t_exact = Instant::now();
    let exact_forest = bp_solve_capped_until(
        &core,
        &terminate,
        Some(Instant::now() + Duration::from_secs(exact_secs)),
    );
    let Some(forest) = exact_forest else {
        println!("  exact  : bp did NOT solve core in {exact_secs}s — skipping trajectory");
        return Ok(());
    };
    // Singleton cut set.
    let mut cut_order: Vec<u32> = forest
        .iter()
        .filter(|c| c.leaves().count() == 1)
        .filter_map(|c| c.leaves().next())
        .collect();
    cut_order.sort_unstable();
    // Multi-leaf agreement components, largest first (contract order).
    let mut comps: Vec<Vec<u32>> = forest
        .iter()
        .map(|c| c.leaves().collect::<Vec<u32>>())
        .filter(|v| v.len() >= 2)
        .collect();
    comps.sort_by_key(|v| std::cmp::Reverse(v.len()));
    println!(
        "  exact  : bp forest {} comps ({} singletons, {} multi-leaf), {:.1}s",
        forest.len(),
        cut_order.len(),
        comps.len(),
        t_exact.elapsed().as_secs_f64()
    );

    let n_cut = cut_order.len().max(1);
    let n_con = comps.len().max(1);

    println!(
        "  {:>6} | {:>30} | {:>30}",
        "commit%", "CONTRACT(must-link): kern/cl/lgSub", "CUT(remove): kern/cl/lgSub"
    );
    for s in 0..=steps {
        let frac = s as f64 / steps as f64;

        let kk = ((n_con as f64) * frac).round() as usize;
        let con_res = residual_contract(&core, &comps, kk.min(comps.len()));
        let (conk, concl, _l, conls) = probe_point(&con_res);

        let kc = ((n_cut as f64) * frac).round() as usize;
        let cut_res = residual_without(&core, &cut_order, kc.min(cut_order.len()));
        let (cutk, cutcl, _l2, cutls) = probe_point(&cut_res);

        println!(
            "  {:>6.0}% | {:>10} {:>4} {:>7} ({:>3.0}%) | {:>10} {:>4} {:>7} ({:>3.0}%)",
            100.0 * frac,
            conk, concl, conls, 100.0 * conls as f64 / core_n as f64,
            cutk, cutcl, cutls, 100.0 * cutls as f64 / core_n as f64,
        );
    }
    println!(
        "  READ: CONTRACT is the faithful B&P test. If its largest_sub (lgSub) drops\n\
         \x20      steeply at LOW commit%, must-link interleaving factorizes the search."
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: decomp_probe <file> [<file> ...]");
        std::process::exit(2);
    }
    let steps: usize = std::env::var("DECOMP_PROBE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seed: u64 = std::env::var("DECOMP_PROBE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    for path in &args {
        if let Err(e) = run_one(path, steps, seed) {
            eprintln!("[{path}] error: {e}");
        }
    }
    Ok(())
}
