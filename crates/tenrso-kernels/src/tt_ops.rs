//! Tensor Train (TT) products and norm operations.
//!
//! This module provides norm computation, inner products, full reconstruction, and
//! matrix-vector products for Tensor Train (TT) format tensors.
//!
//! # Tensor Train Format
//!
//! A d-dimensional tensor X of size n₁ × n₂ × ... × nₐ is represented as:
//!
//! ```text
//! X(i₁, i₂, ..., iₐ) = G₁(i₁) · G₂(i₂) · ... · Gₐ(iₐ)
//! ```
//!
//! where each TT core G_k has shape r_{k-1} × n_k × r_k (with r₀ = rₐ = 1).
//!
//! # Operations
//!
//! ## Products and Norms
//! - [`tt_norm`] - Compute Frobenius norm of TT tensor
//! - [`tt_dot`] - Inner product of two TT tensors
//! - [`tt_to_vector`] - Reconstruct full vector from TT cores
//!
//! ## Matrix Operations
//! - [`tt_matvec`] - TT-matrix times vector product
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray_ext::{Array3, Array2};
//! use tenrso_kernels::tt_ops::*;
//!
//! // Create simple TT cores for a 3D tensor
//! let core1 = Array3::<f64>::ones((1, 4, 2));  // r0=1, n1=4, r1=2
//! let core2 = Array3::<f64>::ones((2, 3, 2));  // r1=2, n2=3, r2=2
//! let core3 = Array3::<f64>::ones((2, 5, 1));  // r2=2, n3=5, r3=1
//! let cores = vec![core1.view(), core2.view(), core3.view()];
//!
//! // Compute TT norm
//! let norm = tt_norm(&cores).unwrap();
//! assert!(norm > 0.0);
//! ```
//!
//! # Performance
//!
//! TT operations are designed for efficiency:
//! - **Norm**: O(∑ᵢ rᵢ³) via core contractions
//! - **Dot product**: O(d · r³) where r is max rank
//!
//! # References
//!
//! - Oseledets, I. V. (2011). "Tensor-Train Decomposition"
//! - Holtz, S., Rohwedder, T., & Schneider, R. (2012). "The Alternating Linear Scheme for Tensor Optimization in the TT Format"

use crate::error::{KernelError, KernelResult};
use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView1, ArrayView3};
use scirs2_core::num_traits::{Float, Num};

/// Compute the Frobenius norm of a TT tensor.
///
/// The norm is computed efficiently by contracting TT cores without full reconstruction.
///
/// # Algorithm
///
/// For TT cores G₁, G₂, ..., Gₐ:
/// 1. Compute Gram matrices: Gₖᵀ · Gₖ for each core
/// 2. Contract these matrices: ||X||² = tr(G₁ᵀG₁ · G₂ᵀG₂ · ... · GₐᵀGₐ)
///
/// # Arguments
///
/// * `cores` - Slice of TT cores, each of shape (r_{k-1}, n_k, r_k)
///
/// # Returns
///
/// Frobenius norm of the TT tensor
///
/// # Complexity
///
/// O(d × r³) where d is the number of cores and r is the maximum rank
///
/// # Errors
///
/// Returns error if cores have incompatible shapes or empty input.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_ops::tt_norm;
///
/// let core1 = Array3::<f64>::ones((1, 4, 2));
/// let core2 = Array3::<f64>::ones((2, 3, 2));
/// let core3 = Array3::<f64>::ones((2, 5, 1));
/// let cores = vec![core1.view(), core2.view(), core3.view()];
///
/// let norm = tt_norm(&cores).unwrap();
/// assert!(norm > 0.0);
/// ```
pub fn tt_norm<T>(cores: &[ArrayView3<T>]) -> KernelResult<T>
where
    T: Float,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("tt_norm", "cores"));
    }

    // Validate core shapes are compatible
    validate_tt_cores(cores)?;

    // Initialize with identity matrix for the boundary condition
    let mut v = Array2::<T>::eye(1);

    // Contract cores from left to right
    for core in cores.iter() {
        v = contract_core_with_v(&v.view(), core)?;
    }

    // Result should be 1x1 matrix containing ||X||²
    if v.shape() != [1, 1] {
        return Err(KernelError::operation_error(
            "tt_norm",
            format!("Expected 1×1 result, got {:?}", v.shape()),
        ));
    }

    let norm_squared = v[[0, 0]];
    if norm_squared < T::zero() {
        return Err(KernelError::operation_error(
            "tt_norm",
            "Negative norm squared",
        ));
    }

    Ok(norm_squared.sqrt())
}

/// Contract a TT core with the running contraction matrix for norm computation.
///
/// For a core G of shape (r_left, n, r_right) and matrix V of shape (r_left, r_left),
/// computes W of shape (r_right, r_right) where:
/// W[i,j] = sum_{alpha,beta,k} V[alpha,beta] * G[alpha,k,i] * G[beta,k,j]
fn contract_core_with_v<T>(
    v: &scirs2_core::ndarray_ext::ArrayView2<T>,
    core: &ArrayView3<T>,
) -> KernelResult<Array2<T>>
where
    T: Float,
{
    let (r_left, n, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);

    if v.shape() != [r_left, r_left] {
        return Err(KernelError::incompatible_shapes(
            "contract_core_with_v",
            vec![r_left, r_left],
            v.shape().to_vec(),
            "Incompatible running matrix shape",
        ));
    }

    let mut w = Array2::<T>::zeros((r_right, r_right));

    for i in 0..r_right {
        for j in 0..r_right {
            let mut sum = T::zero();
            for alpha in 0..r_left {
                for beta in 0..r_left {
                    let v_val = v[[alpha, beta]];
                    for k in 0..n {
                        sum = sum + v_val * core[[alpha, k, i]] * core[[beta, k, j]];
                    }
                }
            }
            w[[i, j]] = sum;
        }
    }

    Ok(w)
}

/// Compute inner product (dot product) of two TT tensors.
///
/// Efficiently computes ⟨X, Y⟩ without full reconstruction by contracting cores.
///
/// # Arguments
///
/// * `cores_x` - TT cores for tensor X
/// * `cores_y` - TT cores for tensor Y
///
/// # Returns
///
/// Inner product ⟨X, Y⟩
///
/// # Complexity
///
/// O(d × r_x × r_y × n × r) where d is depth, r_x, r_y are max ranks
///
/// # Errors
///
/// Returns error if cores have incompatible shapes.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_ops::tt_dot;
///
/// let core1 = Array3::<f64>::ones((1, 3, 2));
/// let core2 = Array3::<f64>::ones((2, 3, 1));
/// let cores_x = vec![core1.view(), core2.view()];
/// let cores_y = cores_x.clone();
///
/// let dot = tt_dot(&cores_x, &cores_y).unwrap();
/// assert!(dot > 0.0);
/// ```
pub fn tt_dot<T>(cores_x: &[ArrayView3<T>], cores_y: &[ArrayView3<T>]) -> KernelResult<T>
where
    T: Float,
{
    if cores_x.is_empty() {
        return Err(KernelError::empty_input("tt_dot", "cores_x"));
    }

    if cores_x.len() != cores_y.len() {
        return Err(KernelError::dimension_mismatch(
            "tt_dot",
            vec![cores_x.len()],
            vec![cores_y.len()],
            "TT tensors must have same depth",
        ));
    }

    // Validate shapes match
    for (cx, cy) in cores_x.iter().zip(cores_y.iter()) {
        if cx.shape()[1] != cy.shape()[1] {
            return Err(KernelError::dimension_mismatch(
                "tt_dot",
                vec![cx.shape()[1]],
                vec![cy.shape()[1]],
                "Core dimensions must match",
            ));
        }
    }

    // Initialize contraction matrix
    let mut result = Array2::<T>::eye(1);

    // Contract core by core
    for (cx, cy) in cores_x.iter().zip(cores_y.iter()) {
        result = contract_cores_for_dot(&result.view(), cx, cy)?;
    }

    // Result should be 1×1 matrix
    if result.shape() != [1, 1] {
        return Err(KernelError::dimension_mismatch(
            "tt_dot",
            vec![1, 1],
            result.shape().to_vec(),
            "Final contraction must produce 1x1 matrix",
        ));
    }

    Ok(result[[0, 0]])
}

/// Reconstruct a full vector from TT cores representing a rank-1 tensor.
///
/// Converts a TT tensor with total dimension n₁ × n₂ × ... × nₐ into a full vector
/// of length n₁ × n₂ × ... × nₐ by contracting all cores.
///
/// # Arguments
///
/// * `cores` - Slice of TT cores, each of shape (r_{k-1}, n_k, r_k)
///
/// # Returns
///
/// A 1D array of length n₁ × n₂ × ... × nₐ containing the full tensor elements
///
/// # Complexity
///
/// O(d × r² × n × N) where d is the number of cores, r is max rank, n is max mode size,
/// and N is the total tensor size
///
/// # Errors
///
/// Returns error if cores have incompatible shapes or empty input.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array3;
/// use tenrso_kernels::tt_ops::tt_to_vector;
///
/// // Create simple TT cores for a 2×3 tensor
/// let core1 = Array3::<f64>::from_shape_fn((1, 2, 2), |(_, i, j)| (i + j) as f64);
/// let core2 = Array3::<f64>::from_shape_fn((2, 3, 1), |(i, j, _)| (i + j) as f64);
/// let cores = vec![core1.view(), core2.view()];
///
/// let vector = tt_to_vector(&cores).unwrap();
/// assert_eq!(vector.len(), 6); // 2 × 3 = 6
/// ```
pub fn tt_to_vector<T>(cores: &[ArrayView3<T>]) -> KernelResult<Array1<T>>
where
    T: Float,
{
    validate_tt_cores(cores)?;

    // Compute total size and mode sizes
    let mode_sizes: Vec<usize> = cores.iter().map(|c| c.shape()[1]).collect();
    let total_size: usize = mode_sizes.iter().product();

    let mut result = Array1::<T>::zeros(total_size);

    // Iterate over all possible index combinations
    for idx in 0..total_size {
        // Convert linear index to multi-index
        let mut multi_idx = vec![0; mode_sizes.len()];
        let mut remaining = idx;
        for (i, &size) in mode_sizes.iter().enumerate().rev() {
            multi_idx[i] = remaining % size;
            remaining /= size;
        }

        // Contract cores for this index
        let mut v = Array2::<T>::ones((1, 1));

        for (k, core) in cores.iter().enumerate() {
            let (r_left, _, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);
            let mode_idx = multi_idx[k];

            // Extract slice for this mode index: core[:, mode_idx, :]
            let core_slice = core.slice(s![.., mode_idx, ..]);

            // Matrix multiplication: v = v @ core_slice
            let mut v_new = Array2::<T>::zeros((v.nrows(), r_right));
            for i in 0..v.nrows() {
                for j in 0..r_right {
                    let mut sum = T::zero();
                    for k_inner in 0..r_left {
                        sum = sum + v[[i, k_inner]] * core_slice[[k_inner, j]];
                    }
                    v_new[[i, j]] = sum;
                }
            }
            v = v_new;
        }

        result[idx] = v[[0, 0]];
    }

    Ok(result)
}

/// Apply a TT-matrix to a vector (TT-matvec operation).
///
/// Multiplies a matrix in TT format by a vector, producing an output vector.
/// The TT matrix is represented by cores where each core Gₖ has shape (r_{k-1}, n_k, m_k, r_k),
/// where n_k are row dimensions and m_k are column dimensions.
///
/// For implementation convenience, cores are provided as 3D arrays of shape (r_{k-1}, n_k × m_k, r_k)
/// with the understanding that n_k × m_k is the combined matrix dimension for mode k.
///
/// # Arguments
///
/// * `cores` - Slice of TT matrix cores
/// * `vector` - Input vector of length ∏ m_k
/// * `row_sizes` - Row dimensions [n₁, n₂, ..., nₐ]
/// * `col_sizes` - Column dimensions [m₁, m₂, ..., mₐ]
///
/// # Returns
///
/// Output vector of length ∏ n_k
///
/// # Complexity
///
/// O(d × r² × n × m) where d is the number of cores, r is max rank,
/// n is max row size, m is max column size
///
/// # Errors
///
/// Returns error if dimensions don't match or cores have incompatible shapes.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::{Array1, Array3};
/// use tenrso_kernels::tt_ops::tt_matvec;
///
/// // Create a simple 2×2 identity-like TT matrix with 2 modes (2×2 total)
/// // Core 1: maps 2 row indices × 2 col indices (4 total) with rank 1→2
/// let core1 = Array3::<f64>::from_shape_fn((1, 4, 2), |(_, ij, k)| {
///     if ij == 0 || ij == 3 { 1.0 } else { 0.0 } // Diagonal pattern
/// });
/// // Core 2: maps 1 row × 1 col (1 total) with rank 2→1
/// let core2 = Array3::<f64>::ones((2, 1, 1));
/// let cores = vec![core1.view(), core2.view()];
///
/// let vector = Array1::from_vec(vec![1.0, 2.0]); // Input vector of length 2
/// let row_sizes = vec![2, 1]; // 2×1 = 2 rows total
/// let col_sizes = vec![2, 1]; // 2×1 = 2 cols total
///
/// let result = tt_matvec(&cores, &vector.view(), &row_sizes, &col_sizes).unwrap();
/// assert_eq!(result.len(), 2);
/// ```
pub fn tt_matvec<T>(
    cores: &[ArrayView3<T>],
    vector: &ArrayView1<T>,
    row_sizes: &[usize],
    col_sizes: &[usize],
) -> KernelResult<Array1<T>>
where
    T: Float,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("tt_matvec", "cores"));
    }

    if cores.len() != row_sizes.len() || cores.len() != col_sizes.len() {
        return Err(KernelError::dimension_mismatch(
            "tt_matvec",
            vec![cores.len()],
            vec![row_sizes.len(), col_sizes.len()],
            "Number of cores must match number of dimensions",
        ));
    }

    let total_cols: usize = col_sizes.iter().product();
    if vector.len() != total_cols {
        return Err(KernelError::dimension_mismatch(
            "tt_matvec",
            vec![total_cols],
            vec![vector.len()],
            "Vector length must match product of column sizes",
        ));
    }

    // Validate that each core's mode size matches row_size * col_size
    for (k, core) in cores.iter().enumerate() {
        let expected_size = row_sizes[k] * col_sizes[k];
        if core.shape()[1] != expected_size {
            return Err(KernelError::dimension_mismatch(
                "tt_matvec",
                vec![expected_size],
                vec![core.shape()[1]],
                format!("Core {} mode size must be row_size × col_size", k),
            ));
        }
    }

    let total_rows: usize = row_sizes.iter().product();
    let mut result = Array1::<T>::zeros(total_rows);

    // Iterate over all row indices
    for row_idx in 0..total_rows {
        // Convert linear row index to multi-index
        let mut row_multi_idx = vec![0; row_sizes.len()];
        let mut remaining = row_idx;
        for (i, &size) in row_sizes.iter().enumerate().rev() {
            row_multi_idx[i] = remaining % size;
            remaining /= size;
        }

        // For each column index, accumulate the product
        let mut sum = T::zero();
        for col_idx in 0..total_cols {
            // Convert linear column index to multi-index
            let mut col_multi_idx = vec![0; col_sizes.len()];
            let mut remaining_col = col_idx;
            for (i, &size) in col_sizes.iter().enumerate().rev() {
                col_multi_idx[i] = remaining_col % size;
                remaining_col /= size;
            }

            // Contract cores for this (row, col) pair
            let mut v = Array2::<T>::ones((1, 1));

            for (k, core) in cores.iter().enumerate() {
                let (r_left, _, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);

                // Compute combined index: row_idx * col_size + col_idx
                let combined_idx = row_multi_idx[k] * col_sizes[k] + col_multi_idx[k];

                // Extract slice for this combined index
                let core_slice = core.slice(s![.., combined_idx, ..]);

                // Matrix multiplication: v = v @ core_slice
                let mut v_new = Array2::<T>::zeros((v.nrows(), r_right));
                for i in 0..v.nrows() {
                    for j in 0..r_right {
                        let mut inner_sum = T::zero();
                        for k_inner in 0..r_left {
                            inner_sum = inner_sum + v[[i, k_inner]] * core_slice[[k_inner, j]];
                        }
                        v_new[[i, j]] = inner_sum;
                    }
                }
                v = v_new;
            }

            sum = sum + v[[0, 0]] * vector[col_idx];
        }

        result[row_idx] = sum;
    }

    Ok(result)
}

/// Contract cores for dot product computation.
fn contract_cores_for_dot<T>(
    prev: &scirs2_core::ndarray_ext::ArrayView2<T>,
    core_x: &ArrayView3<T>,
    core_y: &ArrayView3<T>,
) -> KernelResult<Array2<T>>
where
    T: Float,
{
    let (rx_prev, ry_prev) = (prev.nrows(), prev.ncols());
    let (rx_left, n, rx_right) = (core_x.shape()[0], core_x.shape()[1], core_x.shape()[2]);
    let (ry_left, _, ry_right) = (core_y.shape()[0], core_y.shape()[1], core_y.shape()[2]);

    if rx_left != rx_prev || ry_left != ry_prev {
        return Err(KernelError::dimension_mismatch(
            "contract_cores_for_dot",
            vec![rx_prev, ry_prev],
            vec![rx_left, ry_left],
            "Incompatible core ranks",
        ));
    }

    let mut result = Array2::<T>::zeros((rx_right, ry_right));

    for i in 0..rx_right {
        for j in 0..ry_right {
            let mut sum = T::zero();
            for k in 0..n {
                for ix in 0..rx_left {
                    for iy in 0..ry_left {
                        sum = sum + prev[[ix, iy]] * core_x[[ix, k, i]] * core_y[[iy, k, j]];
                    }
                }
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Validate TT cores have compatible shapes.
///
/// Checks that consecutive cores have matching ranks: r_k of core k equals r_{k-1} of core k+1.
pub(crate) fn validate_tt_cores<T>(cores: &[ArrayView3<T>]) -> KernelResult<()>
where
    T: Num,
{
    if cores.is_empty() {
        return Err(KernelError::empty_input("validate_tt_cores", "cores"));
    }

    // Check first core has r_left = 1
    if cores[0].shape()[0] != 1 {
        return Err(KernelError::dimension_mismatch(
            "validate_tt_cores",
            vec![1],
            vec![cores[0].shape()[0]],
            "First core must have r_left=1",
        ));
    }

    // Check last core has r_right = 1
    if cores[cores.len() - 1].shape()[2] != 1 {
        return Err(KernelError::dimension_mismatch(
            "validate_tt_cores",
            vec![1],
            vec![cores[cores.len() - 1].shape()[2]],
            "Last core must have r_right=1",
        ));
    }

    // Check consecutive cores have matching ranks
    for i in 0..cores.len() - 1 {
        let r_right = cores[i].shape()[2];
        let r_left_next = cores[i + 1].shape()[0];

        if r_right != r_left_next {
            return Err(KernelError::dimension_mismatch(
                "validate_tt_cores",
                vec![r_right],
                vec![r_left_next],
                format!("Rank mismatch between cores {} and {}", i, i + 1),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::{Array1, Array2, Array3};

    #[test]
    fn test_tt_norm_simple() {
        // Create simple TT with known norm
        let core1 = Array3::from_shape_fn((1, 2, 2), |(_, i, j)| if i == j { 1.0 } else { 0.0 });
        let core2 = Array3::from_shape_fn((2, 2, 1), |(i, j, _)| if i == j { 1.0 } else { 0.0 });

        let cores = vec![core1.view(), core2.view()];
        let norm = tt_norm(&cores).unwrap();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_tt_norm_ones() {
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 3, 2));
        let core3 = Array3::<f64>::ones((2, 3, 1));

        let cores = vec![core1.view(), core2.view(), core3.view()];
        let norm = tt_norm(&cores).unwrap();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_tt_dot_self() {
        let core1 = Array3::<f64>::ones((1, 2, 2));
        let core2 = Array3::<f64>::ones((2, 2, 1));

        let cores = vec![core1.view(), core2.view()];
        let dot = tt_dot(&cores, &cores).unwrap();
        let norm = tt_norm(&cores).unwrap();

        // ⟨X, X⟩ should equal ||X||²
        assert!((dot - norm * norm).abs() < 1e-10);
    }

    #[test]
    fn test_tt_dot_orthogonal() {
        // Create two different TT tensors
        let core1_x = Array3::from_shape_fn((1, 2, 2), |(_, i, j)| (i + j + 1) as f64);
        let core2_x = Array3::from_shape_fn((2, 2, 1), |(i, j, _)| (i * 2 + j + 1) as f64);

        let core1_y = Array3::from_shape_fn((1, 2, 2), |(_, i, j)| (i * j + 1) as f64);
        let core2_y = Array3::from_shape_fn((2, 2, 1), |(i, j, _)| ((i + j) * 2 + 1) as f64);

        let cores_x = vec![core1_x.view(), core2_x.view()];
        let cores_y = vec![core1_y.view(), core2_y.view()];

        let dot = tt_dot(&cores_x, &cores_y).unwrap();
        assert!(dot.is_finite());
    }

    #[test]
    fn test_contract_core_with_v() {
        let v = Array2::eye(2);
        let core = Array3::from_shape_fn((2, 3, 2), |(i, j, k)| (i + j + k) as f64);
        let w = contract_core_with_v(&v.view(), &core.view()).unwrap();

        assert_eq!(w.shape(), &[2, 2]);
        // Result should be symmetric for identity V
        assert!((w[[0, 1]] - w[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_tt_to_vector_simple() {
        // Create simple TT cores for a 2×3 tensor
        let core1 = Array3::<f64>::from_shape_fn((1, 2, 2), |(_, i, j)| {
            if i == 0 && j == 0 {
                1.0
            } else if i == 0 && j == 1 {
                2.0
            } else if i == 1 && j == 0 {
                3.0
            } else {
                4.0
            }
        });
        let core2 = Array3::<f64>::from_shape_fn((2, 3, 1), |(i, j, _)| (i + j + 1) as f64);
        let cores = vec![core1.view(), core2.view()];

        let vector = tt_to_vector(&cores).unwrap();
        assert_eq!(vector.len(), 6); // 2 × 3 = 6

        // All values should be non-negative
        for &val in vector.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_tt_to_vector_ones() {
        // TT representation of all-ones tensor
        let core1 = Array3::<f64>::ones((1, 3, 2));
        let core2 = Array3::<f64>::ones((2, 4, 1));
        let cores = vec![core1.view(), core2.view()];

        let vector = tt_to_vector(&cores).unwrap();
        assert_eq!(vector.len(), 12); // 3 × 4 = 12

        // Each element should be the product: 1 * 1 * ... (through ranks)
        // With all-ones cores and proper ranks, each element = 2.0
        for &val in vector.iter() {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tt_to_vector_empty() {
        let cores: Vec<ArrayView3<f64>> = vec![];
        let result = tt_to_vector(&cores);
        assert!(result.is_err());
    }

    #[test]
    fn test_tt_matvec_identity() {
        // Create a simple 2×2 identity-like TT matrix
        // with 2 modes: (2 rows × 2 cols) total
        // Core 1: maps 2×2=4 combined indices, rank 1→2
        let core1 = Array3::<f64>::from_shape_fn((1, 4, 2), |(_, ij, _k)| {
            // Diagonal pattern: ij=0 (row=0,col=0), ij=3 (row=1,col=1)
            if ij == 0 || ij == 3 {
                1.0
            } else {
                0.0
            }
        });
        // Core 2: maps 1×1=1 combined index, rank 2→1
        let core2 = Array3::<f64>::from_elem((2, 1, 1), 0.5);
        let cores = vec![core1.view(), core2.view()];

        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let row_sizes = vec![2, 1];
        let col_sizes = vec![2, 1];

        let result = tt_matvec(&cores, &vector.view(), &row_sizes, &col_sizes).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_tt_matvec_dimension_mismatch() {
        let core1 = Array3::<f64>::ones((1, 4, 2));
        let core2 = Array3::<f64>::ones((2, 1, 1));
        let cores = vec![core1.view(), core2.view()];

        // Wrong vector length
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Should be 2
        let row_sizes = vec![2, 1];
        let col_sizes = vec![2, 1];

        let result = tt_matvec(&cores, &vector.view(), &row_sizes, &col_sizes);
        assert!(result.is_err());
    }

    #[test]
    fn test_tt_matvec_wrong_core_size() {
        let core1 = Array3::<f64>::ones((1, 3, 2)); // Should be 4 not 3
        let core2 = Array3::<f64>::ones((2, 1, 1));
        let cores = vec![core1.view(), core2.view()];

        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let row_sizes = vec![2, 1]; // row_size[0] * col_size[0] should equal core1.shape()[1]
        let col_sizes = vec![2, 1];

        let result = tt_matvec(&cores, &vector.view(), &row_sizes, &col_sizes);
        assert!(result.is_err());
    }

    #[test]
    fn test_tt_matvec_empty_cores() {
        let cores: Vec<ArrayView3<f64>> = vec![];
        let vector = Array1::from_vec(vec![1.0]);
        let row_sizes = vec![];
        let col_sizes = vec![];

        let result = tt_matvec(&cores, &vector.view(), &row_sizes, &col_sizes);
        assert!(result.is_err());
    }
}
