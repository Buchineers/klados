//! Decomposition primitives — utilities that split an instance for the solvers,
//! but are not solvers themselves (no [`crate::Solver`] impl, not in the
//! catalog). Cluster decomposition / reduction and kernelization live in
//! `klados-core`; this module holds the Whidden strict-cluster decomposition.

pub mod whidden_cluster;
