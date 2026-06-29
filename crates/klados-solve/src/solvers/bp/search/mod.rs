//! Search-loop primitives: branching state, search state, selection, telemetry.

pub mod branchings;
pub mod selection;
pub mod state;
pub mod telemetry;

pub use branchings::{Branchings, LeafPair};
pub use selection::{AnySelector, BranchSelector, MostFractionalPair, SelectionContext};
pub use state::{Incumbent, SearchState};
pub use telemetry::{Telemetry, Timings};
