//! Derive RMP column bounds from leaf-pair branching state.
//!
//! A column is forced to 0 if its labelset violates any branching constraint.
//! Otherwise its bound is `[0, ∞)` (or `[1, 1]` if some prior decision pinned
//! it; we don't pin in stage 2, so just `0` or `[0, ∞)`).

use crate::bp::column::AfColumn;
use crate::bp::search::Branchings;

/// Bound for a single column.
#[derive(Clone, Copy)]
pub struct Bound {
    pub lo: f64,
    pub hi: f64,
}

impl Bound {
    pub const FREE: Bound = Bound {
        lo: 0.0,
        hi: f64::INFINITY,
    };
    pub const ZERO: Bound = Bound { lo: 0.0, hi: 0.0 };
}

pub fn bound_for(column: &AfColumn, branchings: &Branchings) -> Bound {
    if branchings.forbids(column) {
        Bound::ZERO
    } else {
        Bound::FREE
    }
}
