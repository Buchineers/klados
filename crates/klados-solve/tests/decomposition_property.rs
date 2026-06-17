//! Property test battery for the decomposition pipeline.
//!
//! For random small 2-tree instances, compute the optimal MAF size via the
//! brute-force oracle, run the production solver (B&P-Multi), and assert:
//!   (a) the solver's component count equals the oracle's optimum;
//!   (b) the solver's output passes the AF validator.
//!
//! This is the gate for any decomposition / kernelization change. If it
//! turns red, do NOT advance to the next phase.
//!
//! Coverage:
//!   - n in {5, 6, 7, 8}, 30 random tree-pairs each (~120 instances)
//!   - Deterministic LCG seed so failures reproduce
//!   - Each instance runs in < 50ms via brute force + B&P; full battery < 30s

use klados_core::Instance;
use klados_core::af_validator::{AfValidation, validate_agreement_forest};
use klados_core::brute_maf::brute_force_maf;
use klados_core::tree::{Label, NONE, NodeId, Tree};
use klados_solve::{RunConfig, Solver};

/// Tiny LCG; we don't need crypto-quality, just reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEFCAFEBABE)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Build a random binary phylogeny on labels 1..=n by repeatedly merging two
/// random "live" subtrees. Equivalent to picking a uniformly random shape over
/// the full set of (2n-3)!! ordered binary trees, but biased by the merge
/// process — good enough for property testing.
fn random_tree(n: u32, rng: &mut Lcg) -> Tree {
    let mut t = Tree::with_capacity(n);
    let mut live: Vec<NodeId> = (1..=n)
        .map(|lbl| {
            let id = t.parent.len() as NodeId;
            t.parent.push(NONE);
            t.left.push(NONE);
            t.right.push(NONE);
            t.label.push(lbl as Label);
            t.label_to_node[lbl as usize] = id;
            id
        })
        .collect();
    while live.len() > 1 {
        let i = rng.range(live.len());
        let a = live.swap_remove(i);
        let j = rng.range(live.len());
        let b = live.swap_remove(j);
        let id = t.parent.len() as NodeId;
        t.parent.push(NONE);
        t.left.push(a);
        t.right.push(b);
        t.label.push(0);
        t.parent[a as usize] = id;
        t.parent[b as usize] = id;
        live.push(id);
    }
    t.root = live[0];
    t.compute_metadata();
    t
}

fn run_battery(n: u32, num_pairs: usize, base_seed: u64) {
    let mut failures: Vec<String> = Vec::new();
    for trial in 0..num_pairs {
        let seed = base_seed.wrapping_add(trial as u64);
        let mut rng = Lcg::new(seed);
        let t1 = random_tree(n, &mut rng);
        let t2 = random_tree(n, &mut rng);
        let inst = Instance::new(vec![t1, t2], n);

        let oracle = brute_force_maf(&inst).expect("brute force should run for small n");

        let mut solver = klados_solve::solvers::maf_branch_price_multi::MafBranchPriceMultiSolver::new();
        let solver_result = solver.solve(&inst, &RunConfig::default());
        let comps = match solver_result {
            Some(c) => c,
            None => {
                failures.push(format!(
                    "n={} seed={}: solver returned None (oracle={})",
                    n, seed, oracle.num_components
                ));
                continue;
            }
        };

        // (b) AF validity
        let validation = validate_agreement_forest(&inst, &comps);
        if !validation.is_ok() {
            failures.push(format!(
                "n={} seed={}: solver output is not a valid AF: {:?} (count={}, oracle={})",
                n,
                seed,
                validation,
                comps.len(),
                oracle.num_components
            ));
            continue;
        }

        // (a) Optimality
        if comps.len() != oracle.num_components {
            failures.push(format!(
                "n={} seed={}: solver={} oracle={} (oracle partition={:?})",
                n,
                seed,
                comps.len(),
                oracle.num_components,
                oracle.partition
            ));
        }
    }

    if !failures.is_empty() {
        let msg = format!(
            "Property battery FAILED: {} failures out of {} trials at n={}\n{}",
            failures.len(),
            num_pairs,
            n,
            failures.join("\n")
        );
        panic!("{}", msg);
    }
}

#[test]
fn property_n5_30trials() {
    run_battery(5, 30, 0xA0A0_0000);
}

#[test]
fn property_n6_30trials() {
    run_battery(6, 30, 0xB0B0_0000);
}

#[test]
fn property_n7_30trials() {
    run_battery(7, 30, 0xC0C0_0000);
}

#[test]
fn property_n8_20trials() {
    run_battery(8, 20, 0xD0D0_0000);
}

#[test]
fn property_n9_15trials() {
    run_battery(9, 15, 0xE0E0_0000);
}

#[test]
fn property_n10_10trials() {
    run_battery(10, 10, 0xF0F0_0000);
}
