//! Branching state: leaf-pair decisions.
//!
//! All branching is on **leaf pairs**, never on column ids. This decouples
//! the search tree from the column pool: ids can grow append-only without
//! invalidating any branchings, children inherit the parent's pool intact,
//! and "stale column id" bugs become impossible.
//!
//! For a column with leafset `L`:
//! - It violates a `must_link {a,b}` iff `|L ∩ {a,b}| == 1`.
//! - It violates a `cannot_link {a,b}` iff `{a,b} ⊆ L`.

use crate::solvers::bp::column::AfColumn;

/// Canonicalized leaf pair (`a < b`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LeafPair {
    pub a: u32,
    pub b: u32,
}

impl LeafPair {
    pub fn new(x: u32, y: u32) -> Self {
        debug_assert!(x != y);
        if x < y {
            Self { a: x, b: y }
        } else {
            Self { a: y, b: x }
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Branchings {
    must_link: Vec<LeafPair>,
    cannot_link: Vec<LeafPair>,
}

impl Branchings {
    pub fn must_link(&self) -> &[LeafPair] {
        &self.must_link
    }

    pub fn cannot_link(&self) -> &[LeafPair] {
        &self.cannot_link
    }

    pub fn depth(&self) -> usize {
        self.must_link.len() + self.cannot_link.len()
    }

    /// Add a must-link constraint (no-op if already present).
    pub fn push_must_link(&mut self, pair: LeafPair) {
        if !self.must_link.contains(&pair) {
            self.must_link.push(pair);
        }
    }

    /// Add a cannot-link constraint (no-op if already present).
    pub fn push_cannot_link(&mut self, pair: LeafPair) {
        if !self.cannot_link.contains(&pair) {
            self.cannot_link.push(pair);
        }
    }

    /// Two children: `(must_link extended, cannot_link extended)`.
    pub fn split_on(&self, pair: LeafPair) -> (Self, Self) {
        let mut left = self.clone();
        if !left.must_link.contains(&pair) {
            left.must_link.push(pair);
        }
        let mut right = self.clone();
        if !right.cannot_link.contains(&pair) {
            right.cannot_link.push(pair);
        }
        (left, right)
    }

    /// True if there is no leafset satisfying every constraint — implied
    /// when must-link and cannot-link conflict on the same pair.
    pub fn is_inconsistent(&self) -> bool {
        for ml in &self.must_link {
            if self.cannot_link.contains(ml) {
                return true;
            }
        }
        false
    }

    /// True if `column` is forbidden by these branchings — i.e., its label
    /// set violates at least one must-link or cannot-link constraint.
    pub fn forbids(&self, column: &AfColumn) -> bool {
        // Hot path on the root node and shallow branches: with no constraints
        // every column is feasible, so skip the binary searches entirely.
        if self.must_link.is_empty() && self.cannot_link.is_empty() {
            return false;
        }
        for ml in &self.must_link {
            let has_a = column.labels().binary_search(&ml.a).is_ok();
            let has_b = column.labels().binary_search(&ml.b).is_ok();
            if has_a != has_b {
                return true;
            }
        }
        for cl in &self.cannot_link {
            if column.labels().binary_search(&cl.a).is_ok()
                && column.labels().binary_search(&cl.b).is_ok()
            {
                return true;
            }
        }
        false
    }
}
