use klados_core::Instance;

fn default_lb_method() -> &'static str {
    option_env!("KLADOS_LB_METHOD").unwrap_or("maf-bounds")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let instance = Instance::from_stdin()?;

    let lb = match default_lb_method() {
        "maf-bounds" => {
            let bounds = klados_core::lower_bound::maf_bounds(
                &instance.trees,
                instance.num_leaves,
            );
            bounds.lower
        }
        "red-blue" => {
            let result = klados_core::lower_bound::red_blue_approx_detailed(
                &instance.trees[0],
                &instance.trees[1],
            );
            result.dual_lb + 1
        }
        "chen-app1" => {
            if instance.num_trees() != 2 {
                eprintln!("klados-lb: chen-app1 requires exactly 2 trees, got {}", instance.num_trees());
                std::process::exit(1);
            }
            let (lb, _ub) = klados_exact::chen_rspr::chen_app1_bounds(
                &instance.trees[0],
                &instance.trees[1],
            );
            lb + 1
        }
        "chen-pair" => {
            if instance.num_trees() != 2 {
                eprintln!("klados-lb: chen-pair requires exactly 2 trees, got {}", instance.num_trees());
                std::process::exit(1);
            }
            let (lb, _ub) = klados_exact::chen_rspr::chen_pair_bounds(
                &instance.trees[0],
                &instance.trees[1],
            );
            lb + 1
        }
        _ => {
            eprintln!("unknown lower bound method: {}", default_lb_method());
            std::process::exit(1);
        }
    };

    println!("{}", lb);
    Ok(())
}
