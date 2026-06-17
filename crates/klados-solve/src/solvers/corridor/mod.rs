//! Reduced-cost corridor solver.
//!
//! After root column generation converges with LP value `L` and
//! incumbent `U`, by LP duality every column in any improving integer
//! solution has reduced cost `≤ γ := U − 1 − L` under the converged
//! duals (proof in the project documentation). The corridor solver
//! enumerates that set completely and solves an exact MIP over it —
//! no branch-and-price tree, no internal time limits.
//!
//! Submodules:
//!
//! * [`topk_m2`] — top-K threshold pair-DP for the m=2 oracle.
//! * [`legacy_anchor_best`] — earlier v1 of the corridor solver kept
//!   for reference and as the registered `corridor` solver name until
//!   the top-K version is wired in.

pub mod lazy_kbest;
pub mod legacy_anchor_best;
pub mod topk_m2;

pub use legacy_anchor_best::CorridorSolver;
