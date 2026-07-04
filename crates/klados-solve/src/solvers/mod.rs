//! The solvers themselves — every module here implements [`crate::Solver`] and
//! is enumerated by [`crate::catalog`]. Non-solver utilities live elsewhere
//! (decomposition primitives in [`crate::decomp`]; kernelization / lower bounds
//! in `klados-core`).

// ── exact-origin solvers ────────────────────────────────────────────────────
pub mod bp;
pub mod chen_rspr;
pub mod collapse;
pub mod corridor;
pub mod maf_branch_price_multi;
pub mod maf_ilp;
pub mod maf_sat;
pub mod overlay_exchange;
pub mod root_corridor;
pub mod root_pool;
pub mod whidden;

// ── heuristic-origin solvers ─────────────────────────────────────────────────
pub mod agglomerative;
pub mod lagrangian;
pub mod max_sat;
pub mod partition;

// ── lower-bound track racer ──────────────────────────────────────────────────
pub mod lower;

// ── feature-based dispatch (portfolio-by-selection) ──────────────────────────
pub mod dispatch;
