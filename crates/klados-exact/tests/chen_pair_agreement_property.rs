//! Property tests for `chen_pair_agreement`.
//!
//! For random small 2-tree instances, assert that:
//!   1. The returned leaf-sets form a valid agreement forest of (T1, T2).
//!   2. The reported `lower` and `upper` bracket the true rSPR distance:
//!         lower <= OPT_cuts <= upper <= 2 * lower
//!   3. The leaf-sets correspond to a partition with `upper + 1` components
//!      (or fewer — padding may happen but should not introduce extras
//!      beyond what the AF requires).

use fixedbitset::FixedBitSet;
use klados_core::Instance;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::brute_maf::brute_force_maf;
use klados_core::tree::{Label, NONE, NodeId, Tree};
use klados_exact::chen_rspr::chen_pair_agreement;

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

fn leafset_of(labels: &[u32], n: u32) -> FixedBitSet {
    let mut bs = FixedBitSet::with_capacity(n as usize + 1);
    for &l in labels {
        bs.insert(l as usize);
    }
    bs
}

fn run_battery(n: u32, num_pairs: usize, base_seed: u64) {
    let mut failures: Vec<String> = Vec::new();
    for trial in 0..num_pairs {
        let seed = base_seed.wrapping_add(trial as u64);
        let mut rng = Lcg::new(seed);
        let t1 = random_tree(n, &mut rng);
        let t2 = random_tree(n, &mut rng);
        let inst = Instance::new(vec![t1.clone(), t2.clone()], n);
        let oracle = brute_force_maf(&inst).expect("brute force should run for small n");
        let opt_comps = oracle.num_components;
        let opt_cuts = opt_comps.saturating_sub(1);

        let (lower, upper, leafsets) = chen_pair_agreement(&t1, &t2);

        // Bounds sanity.
        if !(lower <= opt_cuts && opt_cuts <= upper) {
            failures.push(format!(
                "n={} seed={}: bound violation: lower={} opt_cuts={} upper={}",
                n, seed, lower, opt_cuts, upper
            ));
            continue;
        }
        // 2-approx guarantee: upper <= 2 * lower (when lower > 0).
        if lower > 0 && upper > 2 * lower {
            failures.push(format!(
                "n={} seed={}: 2-approx violated: lower={} upper={}",
                n, seed, lower, upper
            ));
            continue;
        }

        // Convert leafsets → component trees and validate as an AF.
        let comps: Vec<Tree> = leafsets
            .iter()
            .map(|labels| {
                let bs = leafset_of(labels, n);
                Tree::component_from_leafset(&bs, &t1, n)
            })
            .collect();
        let validation = validate_agreement_forest(&inst, &comps);
        if !validation.is_ok() {
            failures.push(format!(
                "n={} seed={}: leaf-sets are not a valid AF: {:?} (count={}, leafsets={:?})",
                n,
                seed,
                validation,
                leafsets.len(),
                leafsets
            ));
            continue;
        }

        // The extracted partition should not be larger than the upper-bound
        // promises (allowing a padding-singleton fudge would weaken the
        // 2-approx guarantee on the partition itself).
        if leafsets.len() > upper + 1 {
            failures.push(format!(
                "n={} seed={}: partition larger than upper bound: comps={} upper+1={}",
                n,
                seed,
                leafsets.len(),
                upper + 1
            ));
            continue;
        }
    }

    if !failures.is_empty() {
        let msg = format!(
            "Chen pair-agreement battery FAILED: {} / {} trials at n={}\n{}",
            failures.len(),
            num_pairs,
            n,
            failures.join("\n")
        );
        panic!("{}", msg);
    }
}

#[test]
fn chen_pair_agreement_n5() {
    run_battery(5, 50, 0xCEA5_0000);
}

#[test]
fn chen_pair_agreement_n6() {
    run_battery(6, 50, 0xCEA6_0000);
}

#[test]
fn chen_pair_agreement_n7() {
    run_battery(7, 50, 0xCEA7_0000);
}

#[test]
fn chen_pair_agreement_n8() {
    run_battery(8, 30, 0xCEA8_0000);
}

#[test]
fn chen_pair_agreement_n10() {
    run_battery(10, 20, 0xCEAA_0000);
}
