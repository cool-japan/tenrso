//! CP-ALS (Canonical Polyadic decomposition via Alternating Least Squares)
//!
//! The CP decomposition factorizes a tensor X into a sum of rank-1 tensors:
//!
//! X ~ Sum_r lambda_r (u_1r x u_2r x ... x u_nr)
//!
//! Where:
//! - R is the CP rank
//! - lambda_r are weights (optional, can be absorbed into factors)
//! - u_ir are factor vectors forming factor matrices U_i in R^(I_i x R)
//!
//! The ALS algorithm alternates between updating each factor matrix while
//! keeping others fixed using MTTKRP and solving a least-squares problem.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Linear algebra operations use `scirs2_linalg`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

mod advanced;
mod core;
pub(crate) mod helpers;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use advanced::*;
pub use core::*;
pub use types::*;
