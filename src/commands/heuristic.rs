use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use klados_core::Instance;
use pace26io::newick::NewickWriter;

static SOLVER_PTR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static SOLVER_VTABLE: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static SIGNALLED: AtomicBool = AtomicBool::new(false);

pub fn run(
    instance: &Instance,
    solver_name: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut solver = klados_heuristic::solver_by_name(solver_name)
        .unwrap_or_else(|| panic!("Unknown heuristic solver: {}", solver_name));

    let solver_ref: &dyn klados_heuristic::HeuristicSolver = &*solver;
    let fat_ptr: (*const (), *const ()) = unsafe { std::mem::transmute(solver_ref) };
    SOLVER_PTR.store(fat_ptr.0 as *mut (), Ordering::SeqCst);
    SOLVER_VTABLE.store(fat_ptr.1 as *mut (), Ordering::SeqCst);
    SIGNALLED.store(false, Ordering::SeqCst);

    let sig_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGTERM, || {
            SIGNALLED.store(true, Ordering::SeqCst);
            let data = SOLVER_PTR.load(Ordering::SeqCst);
            let vtable = SOLVER_VTABLE.load(Ordering::SeqCst);
            if !data.is_null() && !vtable.is_null() {
                let solver: &dyn klados_heuristic::HeuristicSolver =
                    std::mem::transmute((data, vtable));
                solver.sigterm_handler();
            }
        })
    }?;

    let components = solver.solve(instance).expect("Failed to find solution");

    SOLVER_PTR.store(std::ptr::null_mut(), Ordering::SeqCst);
    SOLVER_VTABLE.store(std::ptr::null_mut(), Ordering::SeqCst);

    signal_hook::low_level::unregister(sig_id);

    if verbose {
        eprintln!("Solution: {} components", components.len());
    }

    let mut stdout = io::stdout().lock();
    for tree in &components {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }

    if SIGNALLED.load(Ordering::SeqCst) {
        std::process::exit(0);
    }

    Ok(())
}
