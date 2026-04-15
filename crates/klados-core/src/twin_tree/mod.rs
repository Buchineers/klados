//! Twin-pointer forest pair and algorithms for 2-tree rSPR distance.
//!
//! Core data structures:
//! - `TwinForest`: paired SoA forest with O(1) twin pointers between leaves
//! - `UndoMachine`: checkpoint/rollback for branch-and-bound backtracking
//!
//! Algorithms:
//! - `approx2::approx_2_lb`: Olver et al. 2-approximation dual lower bound (O(n²))

pub mod approx2;
pub mod forest;
pub mod undo;
pub mod zobrist;

pub use approx2::approx_2_lb;
pub use forest::{T1, T2, TwinForest};
pub use undo::UndoMachine;
