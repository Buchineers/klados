use std::io::{self, Write};
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

use klados_core::Instance;
use pace26io::newick::NewickWriter;

static SOLVER_PTR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static SOLVER_VTABLE: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static SIGNAL_KIND: AtomicU8 = AtomicU8::new(0);

const SIGNAL_NONE: u8 = 0;
const SIGNAL_TERM: u8 = 1;
const SIGNAL_INT: u8 = 2;

fn request_graceful_shutdown(signal_kind: u8) {
    SIGNAL_KIND.store(signal_kind, Ordering::SeqCst);
    let data = SOLVER_PTR.load(Ordering::SeqCst);
    let vtable = SOLVER_VTABLE.load(Ordering::SeqCst);
    if !data.is_null() && !vtable.is_null() {
        let solver: &dyn klados_heuristic::HeuristicSolver =
            unsafe { std::mem::transmute((data, vtable)) };
        solver.sigterm_handler();
    }
}

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
    SIGNAL_KIND.store(SIGNAL_NONE, Ordering::SeqCst);

    let sigterm_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGTERM, || {
            request_graceful_shutdown(SIGNAL_TERM);
        })
    }?;
    let sigint_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGINT, || {
            request_graceful_shutdown(SIGNAL_INT);
        })
    }?;

    let components = solver.solve(instance).expect("Failed to find solution");

    SOLVER_PTR.store(std::ptr::null_mut(), Ordering::SeqCst);
    SOLVER_VTABLE.store(std::ptr::null_mut(), Ordering::SeqCst);

    signal_hook::low_level::unregister(sigterm_id);
    signal_hook::low_level::unregister(sigint_id);

    let signal_kind = SIGNAL_KIND.load(Ordering::SeqCst);
    let signalled = signal_kind != SIGNAL_NONE;
    if verbose || signalled {
        if signal_kind == SIGNAL_INT {
            eprintln!("Interrupted: {} components", components.len());
        } else if signal_kind == SIGNAL_TERM {
            eprintln!("Graceful stop: {} components", components.len());
        } else {
            eprintln!("Solution: {} components", components.len());
        }
    }

    if signal_kind != SIGNAL_INT {
        let mut stdout = io::stdout().lock();
        for tree in &components {
            tree.cursor().write_newick(&mut stdout)?;
            writeln!(stdout)?;
        }
    }

    if signal_kind == SIGNAL_INT {
        std::process::exit(130);
    }

    if signal_kind == SIGNAL_TERM {
        std::process::exit(0);
    }

    Ok(())
}
