//! Tensor Train decomposition (TT-SVD and TT-rounding)
//!
//! The Tensor Train (TT) decomposition represents an N-way tensor as a sequence
//! of 3-way tensors (TT-cores):
//!
//! X(i₁, i₂, ..., iₙ) = G₁\[i₁\] × G₂\[i₂\] × ... × Gₙ\[iₙ\]
//!
//! Where:
//! - Gₖ is a TT-core with shape (rₖ₋₁, iₖ, rₖ)
//! - r₀ = rₙ = 1 (boundary conditions)
//! - r₁, r₂, ..., rₙ₋₁ are TT-ranks
//!
//! # Algorithms
//!
//! ## TT-SVD
//! Computes TT decomposition via sequential SVD with rank truncation.
//! Time: O(N × I³ × R²) where I = max mode size, R = max TT-rank
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! SVD operations use `scirs2_linalg::decomposition`.
//! Direct use of `ndarray` or `num_traits` is forbidden per SCIRS2_INTEGRATION_POLICY.md
//!
//! # Module Layout
//!
//! - `types` — `TTError`, `TTDecomp`, `TTMatrix`, and `tt_matrix_from_diagonal`
//! - `algorithms` — TT-SVD construction and TT-rounding
//! - `operations` — TT addition, inner product, Hadamard product

mod algorithms;
mod operations;
mod types;

#[cfg(test)]
mod tests;

pub use algorithms::{tt_round, tt_svd};
pub use operations::{tt_add, tt_dot, tt_hadamard};
pub use types::{tt_matrix_from_diagonal, TTDecomp, TTError, TTMatrix};
