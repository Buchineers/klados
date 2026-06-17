//! Decomposition primitives — utilities that split an instance for the solvers
//! but are not solvers themselves. Cluster decomposition / reduction and
//! kernelization live in `klados-core`; this module holds the Whidden
//! strict-cluster decomposition.

pub mod whidden_cluster;
