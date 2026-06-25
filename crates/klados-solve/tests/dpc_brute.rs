//! Validate the coupling-width DP state-counter's μ* against the brute oracle,
//! for every reference tree, on random small multi-tree instances.

use klados_core::Instance;
use klados_core::brute_maf::brute_force_maf;
use klados_core::tree::{Label, NONE, NodeId, Tree};

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

#[test]
fn dpc_matches_brute_all_t0() {
    let mut failures = Vec::new();
    let mut trials = 0;
    for n in [5u32, 6, 7] {
        for m in [2usize, 3] {
            for seed in 0..40u64 {
                let mut rng = Lcg::new(0x515A_0000 ^ (n as u64) << 8 ^ (m as u64) << 4 ^ seed);
                let trees: Vec<Tree> = (0..m).map(|_| random_tree(n, &mut rng)).collect();
                let inst = Instance::new(trees, n);
                let oracle = brute_force_maf(&inst).expect("brute runs");
                let true_mu = (n as usize) - oracle.num_components;
                trials += 1;
                for t0 in 0..m {
                    match klados_solve::solvers::ncpack::dpc_mu_for_t0(&inst, t0, 5_000_000) {
                        Some((_, _, mu)) => {
                            if mu != true_mu {
                                failures.push(format!(
                                    "n={n} m={m} seed={seed} t0={t0}: dp_mu={mu} true_mu={true_mu} (OPT brute={})",
                                    oracle.num_components
                                ));
                            }
                        }
                        None => failures.push(format!("n={n} m={m} seed={seed} t0={t0}: capped")),
                    }
                }
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} / {} (instance,t0) mismatches:\n{}",
            failures.len(),
            trials * 2,
            failures.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}

#[test]
fn debug_seed21() {
    use klados_core::af_validator::canonical_newick;
    let n = 6u32; let m = 2usize; let seed = 21u64;
    let mut rng = Lcg::new(0x515A_0000 ^ (n as u64) << 8 ^ (m as u64) << 4 ^ seed);
    let trees: Vec<Tree> = (0..m).map(|_| random_tree(n, &mut rng)).collect();
    for (i,t) in trees.iter().enumerate() { eprintln!("T{i} = {}", canonical_newick(t)); }
    let inst = Instance::new(trees, n);
    let oracle = brute_force_maf(&inst).unwrap();
    eprintln!("brute: components={} partition={:?}", oracle.num_components, oracle.partition);
    for t0 in 0..m {
        eprintln!("DP t0={t0}: {:?}", klados_solve::solvers::ncpack::dpc_mu_for_t0(&inst, t0, 5_000_000));
    }
}

// Enumerate set partitions of 1..=n
fn partitions(n: u32) -> Vec<Vec<Vec<u32>>> {
    let mut res = vec![];
    fn go(cur: u32, n: u32, parts: &mut Vec<Vec<u32>>, res: &mut Vec<Vec<Vec<u32>>>) {
        if cur > n { res.push(parts.clone()); return; }
        for i in 0..parts.len() {
            parts[i].push(cur); go(cur+1,n,parts,res); parts[i].pop();
        }
        parts.push(vec![cur]); go(cur+1,n,parts,res); parts.pop();
    }
    go(1,n,&mut vec![],&mut res); res
}

#[test]
fn checks_vs_validator() {
    use klados_core::af_validator::{validate_agreement_forest, AfValidation};
    let n=6u32; let m=2usize; let seed=21u64;
    let mut rng = Lcg::new(0x515A_0000 ^ (n as u64) << 8 ^ (m as u64) << 4 ^ seed);
    let trees: Vec<Tree> = (0..m).map(|_| random_tree(n, &mut rng)).collect();
    let inst = Instance::new(trees.clone(), n);
    let mut disagree=0;
    for part in partitions(n) {
        // build component trees via prune
        let comps: Vec<Tree> = part.iter().map(|b| {
            let mut keep = fixedbitset::FixedBitSet::with_capacity(n as usize+1);
            for &l in b { keep.insert(l as usize); }
            inst.trees[0].prune_to_leafset(&keep)
        }).collect();
        let validator_ok = matches!(validate_agreement_forest(&inst,&comps), AfValidation::Ok);
        // my checks
        let blocks: Vec<&Vec<u32>> = part.iter().collect();
        let my_agree = part.iter().all(|b| klados_solve::solvers::ncpack::dpc_agrees_pub(&inst,b));
        let my_noncross = klados_solve::solvers::ncpack::dpc_noncross_pub(&inst,&blocks);
        let my_ok = my_agree && my_noncross;
        if my_ok != validator_ok {
            disagree+=1;
            if disagree<=10 { eprintln!("DISAGREE part={:?} validator={} my_ok={} (agree={} noncross={})", part, validator_ok, my_ok, my_agree, my_noncross); }
        }
    }
    eprintln!("total disagreements: {disagree}");
    assert_eq!(disagree, 0, "my checks disagree with validator");
}
