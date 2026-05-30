//! Tensor Train (TT) orthogonalization operations.
//!
//! This module provides QR-based orthogonalization for Tensor Train (TT) cores.
//! Left and right orthogonalization are fundamental building blocks for TT rounding,
//! norm computation, and other TT algorithms.
//!
//! # Orthogonalization
//!
//! ## Left-Orthogonalization
//! A TT core G_k of shape (r_{k-1}, n_k, r_k) is **left-orthogonal** if when reshaped
//! to a (r_{k-1} × n_k, r_k) matrix M, it satisfies M^T M = I.
//!
//! ## Right-Orthogonalization
//! A TT core G_k is **right-orthogonal** if the (r_{k-1}, n_k × r_k) reshape M
//! satisfies M M^T = I.
//!
//! # Functions
//!
//! - [`tt_left_orthogonalize_core`] - Left-orthogonalize a single TT core
//! - [`tt_right_orthogonalize_core`] - Right-orthogonalize a single TT core
//! - [`tt_left_orthogonalize`] - Left-orthogonalize all TT cores
//! - [`tt_right_orthogonalize`] - Right-orthogonalize all TT cores
//!
//! # References
//!
//! - Oseledets, I. V. (2011). "Tensor-Train Decomposition"
//! - Holtz, S., Rohwedder, T., & Schneider, R. (2012). "The Alternating Linear Scheme for Tensor Optimization in the TT Format"

use crate::error::{KernelError, KernelResult};
use scirs2_core::ndarray_ext::{Array2, Array3, ArrayView2, ArrayView3, ScalarOperand};
use scirs2_core::num_traits::{Float, NumAssign};
use scirs2_linalg::qr;
use std::iter::Sum;

/// Left-orthogonalize a single TT core using QR decomposition.
///
/// Transforms the core G of shape (r_left × n × r_right) into an orthogonal matrix
/// Q and a remainder matrix R, such that G_reshaped = Q · R.
///
/// # Arguments
///
/// * `core` - Input TT core of shape (r_left, n, r_right)
///
/// # Returns
///
/// * `(Q, R)` where:
///   - Q has shape (r_left × n, r_new) with orthonormal columns
///   - R has shape (r_new, r_right) as the remainder
///
/// # Complexity
///
/// O(r_left × n × r_right × min(r_left × n, r_right))
///
/// # Errors
///
/// Returns error if QR decomposition fails or shapes are invalid.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_orthog::tt_left_orthogonalize_core;
///
/// let core = Array3::<f64>::ones((2, 3, 4));
/// let (q, r) = tt_left_orthogonalize_core(&core.view()).unwrap();
/// assert_eq!(q.shape()[1], r.shape()[0]); // Dimensions match
/// ```
pub fn tt_left_orthogonalize_core<T>(core: &ArrayView3<T>) -> KernelResult<(Array2<T>, Array2<T>)>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (r_left, n, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);

    // Reshape core to matrix (r_left * n, r_right)
    let core_mat = core
        .view()
        .to_shape((r_left * n, r_right))
        .map_err(|e| {
            KernelError::operation_error(
                "tt_left_orthogonalize_core",
                format!("Failed to reshape core: {}", e),
            )
        })?
        .to_owned();

    // Perform QR decomposition using scirs2_core
    // Note: We'll need to implement QR or use scirs2-linalg when available
    // For now, we'll use a simplified orthogonalization via SVD
    tt_qr_decomposition(&core_mat.view())
}

/// Right-orthogonalize a single TT core using QR decomposition.
///
/// Similar to left-orthogonalization but processes from right to left.
///
/// # Arguments
///
/// * `core` - Input TT core of shape (r_left, n, r_right)
///
/// # Returns
///
/// * `(L, Q)` where:
///   - L has shape (r_left, r_new) as the remainder
///   - Q has shape (r_new, n × r_right) with orthonormal rows
///
/// # Complexity
///
/// O(r_left × n × r_right × min(r_left, n × r_right))
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_orthog::tt_right_orthogonalize_core;
///
/// let core = Array3::<f64>::ones((2, 3, 4));
/// let (l, q) = tt_right_orthogonalize_core(&core.view()).unwrap();
/// assert_eq!(l.shape()[1], q.shape()[0]); // Dimensions match
/// ```
pub fn tt_right_orthogonalize_core<T>(core: &ArrayView3<T>) -> KernelResult<(Array2<T>, Array2<T>)>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (r_left, n, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);

    // Reshape core to matrix (r_left, n * r_right)
    let core_mat = core
        .view()
        .to_shape((r_left, n * r_right))
        .map_err(|e| {
            KernelError::operation_error(
                "tt_right_orthogonalize_core",
                format!("Failed to reshape core: {}", e),
            )
        })?
        .to_owned();

    // Perform QR decomposition on transposed matrix
    let core_mat_t = core_mat.t().to_owned();
    let (q_t, r_t) = tt_qr_decomposition(&core_mat_t.view())?;

    // Transpose back
    Ok((r_t.t().to_owned(), q_t.t().to_owned()))
}

/// Perform QR decomposition using scirs2-linalg.
///
/// Handles both tall (m >= n) and wide (m < n) matrices.
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * `(Q, R)` - Orthogonal matrix Q and upper triangular R
pub(crate) fn tt_qr_decomposition<T>(matrix: &ArrayView2<T>) -> KernelResult<(Array2<T>, Array2<T>)>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = (matrix.nrows(), matrix.ncols());

    if m >= n {
        // Standard QR: m >= n
        qr(matrix, None).map_err(|e| {
            KernelError::operation_error("tt_qr_decomposition", format!("QR failed: {}", e))
        })
    } else {
        // Wide matrix: transpose, do QR, transpose back
        let matrix_t = matrix.t().to_owned();
        let (q_t, r_t) = qr(&matrix_t.view(), None).map_err(|e| {
            KernelError::operation_error(
                "tt_qr_decomposition",
                format!("QR on transpose failed: {}", e),
            )
        })?;

        // For A = Q @ R, we have A^T = R^T @ Q^T
        // So A = (A^T)^T = (R^T @ Q^T)^T = Q @ R^T
        // Return Q and R^T
        Ok((q_t.t().to_owned(), r_t.t().to_owned()))
    }
}

/// Left-orthogonalize all TT cores using QR decomposition.
///
/// Processes cores from left to right, making each core left-orthogonal and
/// absorbing the remainder into the next core.
///
/// # Arguments
///
/// * `cores` - Input TT cores (will be modified in-place)
///
/// # Returns
///
/// * `Ok(())` on success
///
/// # Complexity
///
/// O(∑ᵢ rᵢ² nᵢ) where rᵢ and nᵢ are ranks and mode sizes
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_orthog::tt_left_orthogonalize;
///
/// let core1 = Array3::<f64>::ones((1, 3, 2));
/// let core2 = Array3::<f64>::ones((2, 4, 2));
/// let core3 = Array3::<f64>::ones((2, 5, 1));
/// let mut cores = vec![core1, core2, core3];
///
/// tt_left_orthogonalize(&mut cores).unwrap();
/// // Cores are now left-orthogonal
/// ```
pub fn tt_left_orthogonalize<T>(cores: &mut [Array3<T>]) -> KernelResult<()>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("tt_left_orthogonalize", "cores"));
    }

    for i in 0..cores.len() - 1 {
        let (r_left, n, r_right) = {
            let shape = cores[i].shape();
            (shape[0], shape[1], shape[2])
        };

        // Reshape core to matrix (r_left * n, r_right)
        let core_mat = cores[i]
            .view()
            .to_shape((r_left * n, r_right))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_left_orthogonalize",
                    format!("Failed to reshape core {}: {}", i, e),
                )
            })?
            .to_owned();

        // QR decomposition
        let (q, r) = tt_qr_decomposition(&core_mat.view())?;

        // Update current core with Q
        let new_rank = q.ncols();
        cores[i] = q
            .to_shape((r_left, n, new_rank))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_left_orthogonalize",
                    format!("Failed to reshape Q for core {}: {}", i, e),
                )
            })?
            .to_owned();

        // Absorb R into next core
        let next_shape = cores[i + 1].shape();
        let (_, n_next, r_right_next) = (next_shape[0], next_shape[1], next_shape[2]);

        let next_mat = cores[i + 1]
            .view()
            .to_shape((r_right, n_next * r_right_next))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_left_orthogonalize",
                    format!("Failed to reshape next core {}: {}", i + 1, e),
                )
            })?
            .to_owned();

        let updated_next = r.dot(&next_mat);

        cores[i + 1] = updated_next
            .to_shape((new_rank, n_next, r_right_next))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_left_orthogonalize",
                    format!("Failed to reshape updated core {}: {}", i + 1, e),
                )
            })?
            .to_owned();
    }

    Ok(())
}

/// Right-orthogonalize all TT cores using QR decomposition.
///
/// Processes cores from right to left, making each core right-orthogonal and
/// absorbing the remainder into the previous core.
///
/// # Arguments
///
/// * `cores` - Input TT cores (will be modified in-place)
///
/// # Returns
///
/// * `Ok(())` on success
///
/// # Complexity
///
/// O(∑ᵢ rᵢ² nᵢ) where rᵢ and nᵢ are ranks and mode sizes
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_orthog::tt_right_orthogonalize;
///
/// let core1 = Array3::<f64>::ones((1, 3, 2));
/// let core2 = Array3::<f64>::ones((2, 4, 2));
/// let core3 = Array3::<f64>::ones((2, 5, 1));
/// let mut cores = vec![core1, core2, core3];
///
/// tt_right_orthogonalize(&mut cores).unwrap();
/// // Cores are now right-orthogonal
/// ```
pub fn tt_right_orthogonalize<T>(cores: &mut [Array3<T>]) -> KernelResult<()>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("tt_right_orthogonalize", "cores"));
    }

    for i in (1..cores.len()).rev() {
        let (r_left, n, r_right) = {
            let shape = cores[i].shape();
            (shape[0], shape[1], shape[2])
        };

        // Reshape core to matrix (r_left, n * r_right)
        let core_mat = cores[i]
            .view()
            .to_shape((r_left, n * r_right))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_right_orthogonalize",
                    format!("Failed to reshape core {}: {}", i, e),
                )
            })?
            .to_owned();

        // QR decomposition on transposed matrix
        let core_mat_t = core_mat.t().to_owned();
        let (q_t, r_t) = tt_qr_decomposition(&core_mat_t.view())?;

        // Update current core with Q^T
        let new_rank = q_t.nrows();
        let q = q_t.t().to_owned();
        cores[i] = q
            .to_shape((new_rank, n, r_right))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_right_orthogonalize",
                    format!("Failed to reshape Q for core {}: {}", i, e),
                )
            })?
            .to_owned();

        // Absorb R^T into previous core
        let prev_shape = cores[i - 1].shape();
        let (r_left_prev, n_prev, _) = (prev_shape[0], prev_shape[1], prev_shape[2]);

        let prev_mat = cores[i - 1]
            .view()
            .to_shape((r_left_prev * n_prev, r_left))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_right_orthogonalize",
                    format!("Failed to reshape prev core {}: {}", i - 1, e),
                )
            })?
            .to_owned();

        let r = r_t.t().to_owned();
        let updated_prev = prev_mat.dot(&r);

        cores[i - 1] = updated_prev
            .to_shape((r_left_prev, n_prev, new_rank))
            .map_err(|e| {
                KernelError::operation_error(
                    "tt_right_orthogonalize",
                    format!("Failed to reshape updated core {}: {}", i - 1, e),
                )
            })?
            .to_owned();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt_ops::{tt_norm, validate_tt_cores};
    use scirs2_core::ndarray_ext::Array3;

    #[test]
    fn test_left_orthogonalize_core() {
        let core = Array3::<f64>::ones((2, 3, 4));
        let result = tt_left_orthogonalize_core(&core.view());
        assert!(result.is_ok());

        let (q, r) = result.unwrap();
        assert_eq!(q.shape()[1], r.shape()[0]);
    }

    #[test]
    fn test_right_orthogonalize_core() {
        let core = Array3::<f64>::ones((2, 3, 4));
        let result = tt_right_orthogonalize_core(&core.view());
        assert!(result.is_ok());

        let (l, q) = result.unwrap();
        assert_eq!(l.shape()[1], q.shape()[0]);
    }

    #[test]
    fn test_tt_left_orthogonalize() {
        let core1 = Array3::<f64>::from_shape_fn((1, 3, 2), |(_, i, j)| (i + j + 1) as f64);
        let core2 = Array3::<f64>::from_shape_fn((2, 4, 2), |(i, j, k)| (i + j + k + 1) as f64);
        let core3 = Array3::<f64>::from_shape_fn((2, 5, 1), |(i, j, _)| (i + j + 1) as f64);

        let mut cores = vec![core1, core2, core3];
        let result = tt_left_orthogonalize(&mut cores);
        assert!(result.is_ok());

        // Check that shapes are maintained or reduced
        assert_eq!(cores[0].shape()[0], 1);
        assert_eq!(cores[2].shape()[2], 1);
    }

    #[test]
    fn test_tt_right_orthogonalize() {
        let core1 = Array3::<f64>::from_shape_fn((1, 3, 2), |(_, i, j)| (i + j + 1) as f64);
        let core2 = Array3::<f64>::from_shape_fn((2, 4, 2), |(i, j, k)| (i + j + k + 1) as f64);
        let core3 = Array3::<f64>::from_shape_fn((2, 5, 1), |(i, j, _)| (i + j + 1) as f64);

        let mut cores = vec![core1, core2, core3];
        let result = tt_right_orthogonalize(&mut cores);
        assert!(result.is_ok());

        // Check that shapes are maintained or reduced
        assert_eq!(cores[0].shape()[0], 1);
        assert_eq!(cores[2].shape()[2], 1);
    }

    #[test]
    fn test_tt_orthogonalize_preserves_norm() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 3, 2));
        let core3 = Array3::<f64>::ones((2, 3, 1));

        let cores_ref = vec![core1.view(), core2.view(), core3.view()];
        let original_norm = tt_norm(&cores_ref).unwrap();

        let mut cores_left = vec![core1.clone(), core2.clone(), core3.clone()];
        tt_left_orthogonalize(&mut cores_left).unwrap();
        let cores_left_view: Vec<_> = cores_left.iter().map(|c| c.view()).collect();
        let left_norm = tt_norm(&cores_left_view).unwrap();

        let mut cores_right = vec![core1, core2, core3];
        tt_right_orthogonalize(&mut cores_right).unwrap();
        let cores_right_view: Vec<_> = cores_right.iter().map(|c| c.view()).collect();
        let right_norm = tt_norm(&cores_right_view).unwrap();

        // Norm should be preserved
        assert!((original_norm - left_norm).abs() < 1e-8);
        assert!((original_norm - right_norm).abs() < 1e-8);
    }

    #[test]
    fn test_tt_left_orthogonalize_single_core() {
        let core1 = Array3::<f64>::ones((1, 5, 1));
        let mut cores = vec![core1];
        let result = tt_left_orthogonalize(&mut cores);
        // Should succeed but not change anything (single core)
        assert!(result.is_ok());
        assert_eq!(cores[0].shape(), &[1, 5, 1]);
    }

    #[test]
    fn test_tt_right_orthogonalize_single_core() {
        let core1 = Array3::<f64>::ones((1, 5, 1));
        let mut cores = vec![core1];
        let result = tt_right_orthogonalize(&mut cores);
        // Should succeed but not change anything (single core)
        assert!(result.is_ok());
        assert_eq!(cores[0].shape(), &[1, 5, 1]);
    }

    #[test]
    fn test_validate_tt_cores_valid() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 4, 3));
        let core3 = Array3::<f64>::ones((3, 5, 1));

        let cores = vec![core1.view(), core2.view(), core3.view()];
        assert!(validate_tt_cores(&cores).is_ok());
    }

    #[test]
    fn test_validate_tt_cores_invalid_first() {
        let core1 = Array3::<f64>::ones((2, 3, 2)); // Wrong: should be 1
        let core2 = Array3::<f64>::ones((2, 4, 1));

        let cores = vec![core1.view(), core2.view()];
        assert!(validate_tt_cores(&cores).is_err());
    }

    #[test]
    fn test_validate_tt_cores_invalid_last() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 4, 2)); // Wrong: should be 1

        let cores = vec![core1.view(), core2.view()];
        assert!(validate_tt_cores(&cores).is_err());
    }

    #[test]
    fn test_validate_tt_cores_mismatch() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((3, 4, 1)); // Wrong: should be 2

        let cores = vec![core1.view(), core2.view()];
        assert!(validate_tt_cores(&cores).is_err());
    }
}
