//! Detailed bounds comparison: red-blue (UB + dual LB), Chen 2-approx, Wu LP relax, Olver LP*.

use klados_core::Instance;
use klados_core::lower_bound::{cherry_reduce_ub, maf_bounds, red_blue_approx_detailed};
use klados_exact::chen_rspr::chen_pair_bounds;
use klados_exact::whidden::approx_2_lb_for_instance;

pub fn run(instance: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    let n = instance.num_leaves;
    let m = instance.num_trees();

    eprintln!("Instance: m={} n={}", m, n);

    // 1. Standard greedy bounds (fast).
    let t0 = std::time::Instant::now();
    let bounds = maf_bounds(&instance.trees, n);
    let bounds_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Greedy bounds: LB={} UB={} ({:.1}ms)",
        bounds.lower, bounds.upper, bounds_ms
    );

    // 2. Pairwise red-blue and Chen 2-approx.
    if m == 2 {
        let t0 = std::time::Instant::now();
        let rb = red_blue_approx_detailed(&instance.trees[0], &instance.trees[1]);
        let rb_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let lb_half = (rb.ub + 1) / 2;
        eprintln!(
            "Red-Blue:      UB={} LB_half={} LB_dual={} ({:.1}ms)",
            rb.ub, lb_half, rb.dual_lb, rb_ms
        );

        let t0 = std::time::Instant::now();
        let a2_lb = approx_2_lb_for_instance(&instance.trees[0], &instance.trees[1], n);
        let a2_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Olver-TF LB:   {} ({:.1}ms)", a2_lb, a2_ms);

        let t0 = std::time::Instant::now();
        let (chen_lb, chen_ub) = chen_pair_bounds(&instance.trees[0], &instance.trees[1]);
        let chen_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let chen_lb_comps = chen_lb + 1; // rSPR cuts -> MAF components
        let chen_ub_comps = chen_ub + 1;
        eprintln!(
            "Chen 2-approx: LB_cuts={} UB_cuts={} → LB_comps={} UB_comps={} ({:.1}ms)  ratio={:.3}",
            chen_lb,
            chen_ub,
            chen_lb_comps,
            chen_ub_comps,
            chen_ms,
            chen_ub_comps as f64 / chen_lb_comps.max(1) as f64,
        );

        let t0 = std::time::Instant::now();
        let cherry_ub = cherry_reduce_ub(&instance.trees[0], &instance.trees[1]);
        let cherry_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Cherry UB:     {} ({:.1}ms)", cherry_ub, cherry_ms);
    } else {
        // Multi-tree: pairwise red-blue.
        let mut best_rb_ub = usize::MAX;
        let mut best_rb_dual = 0usize;
        let mut best_rb_half = 0usize;
        let t0 = std::time::Instant::now();
        for i in 0..m {
            for j in (i + 1)..m {
                let rb = red_blue_approx_detailed(&instance.trees[i], &instance.trees[j]);
                let lb_half = (rb.ub + 1) / 2;
                if rb.ub < best_rb_ub {
                    best_rb_ub = rb.ub;
                }
                if rb.dual_lb > best_rb_dual {
                    best_rb_dual = rb.dual_lb;
                }
                if lb_half > best_rb_half {
                    best_rb_half = lb_half;
                }
            }
        }
        let rb_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "Red-Blue pair: best_UB={} best_LB_half={} best_LB_dual={} ({:.1}ms)",
            best_rb_ub, best_rb_half, best_rb_dual, rb_ms
        );
    }

    // 3. Wu LP relaxation (our ILP with continuous vars, C2 ≤ 1).
    if n <= 60 {
        let t0 = std::time::Instant::now();
        if let Some(wu_val) = klados_exact::maf_ilp::solve_lp_relaxation(instance) {
            let wu_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let wu_lb = wu_val.ceil() as usize; // ceil to get integer LB
            eprintln!(
                "Wu LP relax:   {:.4} (ceil={}) ({:.1}ms)",
                wu_val, wu_lb, wu_ms
            );
        }
    } else {
        eprintln!("Wu LP relax:   skipped (n={} > 60)", n);
    }

    // Print summary line for easy parsing.
    println!(
        "m={} n={} greedy_lb={} greedy_ub={}",
        m, n, bounds.lower, bounds.upper
    );

    Ok(())
}
