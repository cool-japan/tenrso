//! Tensor operations: reshape, permute, unfold, fold.
//!
//! This module documents the tensor operations available in tenrso-core.
//! All operations are implemented as methods on [`DenseND`](crate::dense::DenseND).
//!
//! # Available Operations
//!
//! ## Shape Manipulation
//!
//! - **Reshape:** [`DenseND::reshape`](crate::dense::DenseND::reshape)
//!   - Change tensor shape while preserving total number of elements
//!   - Zero-copy when memory layout is contiguous
//!   - Example: `[2, 3, 4]` → `[6, 4]`
//!
//! - **Permute:** [`DenseND::permute`](crate::dense::DenseND::permute)
//!   - Reorder tensor axes (generalized transpose)
//!   - Example: `[2, 3, 4]` with axes `[2, 0, 1]` → `[4, 2, 3]`
//!
//! ## Matricization (for Decompositions)
//!
//! - **Unfold:** [`DenseND::unfold`](crate::dense::DenseND::unfold)
//!   - Matricize tensor along a specified mode
//!   - Critical for CP-ALS, Tucker-HOOI, and other decompositions
//!   - Example: `[2, 3, 4]` unfolded on mode 1 → `[3, 8]` matrix
//!
//! - **Fold:** [`DenseND::fold`](crate::dense::DenseND::fold)
//!   - Inverse of unfold - reconstruct tensor from matrix
//!   - Example: `[3, 8]` matrix folded to `[2, 3, 4]` on mode 1
//!
//! # Examples
//!
//! ## Reshape Operation
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
//! let reshaped = tensor.reshape(&[6, 4]).unwrap();
//! assert_eq!(reshaped.shape(), &[6, 4]);
//! ```
//!
//! ## Permute Operation
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! // Create a 3D tensor
//! let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
//!
//! // Permute: move last axis to front
//! let permuted = tensor.permute(&[2, 0, 1]).unwrap();
//! assert_eq!(permuted.shape(), &[4, 2, 3]);
//! ```
//!
//! ## Unfold/Fold Operations
//!
//! ```
//! use tenrso_core::DenseND;
//!
//! let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
//!
//! // Unfold along mode 1
//! let unfolded = tensor.unfold(1).unwrap();
//! assert_eq!(unfolded.shape(), &[3, 8]); // 8 = 2 * 4
//!
//! // Fold back to original shape
//! let folded = DenseND::fold(&unfolded, &[2, 3, 4], 1).unwrap();
//! assert_eq!(folded.shape(), &[2, 3, 4]);
//! ```
//!
//! # Mathematical Background
//!
//! ## Mode-n Unfolding
//!
//! Mode-n unfolding (also called matricization) rearranges a tensor into a matrix
//! by fixing one mode and concatenating all fibers along that mode.
//!
//! For a tensor **X** ∈ ℝ^(I₁ × I₂ × ... × Iₙ), the mode-n unfolding **X₍ₙ₎**
//! is a matrix of size Iₙ × (I₁ × ... × Iₙ₋₁ × Iₙ₊₁ × ... × Iₙ).
//!
//! ## Applications
//!
//! - **CP Decomposition:** Unfold tensor along each mode for alternating least squares
//! - **Tucker Decomposition:** Compute mode-n SVD on unfolded matrices
//! - **Tensor-Train:** Successive unfolding and reshaping for TT-SVD
//!
//! # Performance Notes
//!
//! - **Reshape:** O(1) for contiguous tensors, O(n) for non-contiguous
//! - **Permute:** O(n) - creates new tensor with permuted strides
//! - **Unfold:** O(n) - requires data rearrangement
//! - **Fold:** O(n) - inverse rearrangement of unfold
//!
//! All operations use scirs2-core's optimized array operations.
