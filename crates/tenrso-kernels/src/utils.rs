//! Utility functions and helpers for tensor kernel operations
//!
//! This module provides convenience functions and performance helpers
//! that simplify common patterns when working with tensor kernels.

use crate::error::{KernelError, KernelResult};
use crate::{khatri_rao, mttkrp};
use scirs2_core::ndarray_ext::{Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::{Float, Num};
use std::time::Instant;

/// Performance timing result for kernel operations
#[derive(Debug, Clone)]
pub struct TimingResult {
    /// Operation name
    pub operation: String,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
    /// Throughput in Gelem/s (if applicable)
    pub throughput_gelem_s: Option<f64>,
    /// Number of elements processed
    pub elements: usize,
}

impl TimingResult {
    /// Create a new timing result
    pub fn new(operation: impl Into<String>, elapsed_ms: f64, elements: usize) -> Self {
        let throughput_gelem_s = if elapsed_ms > 0.0 {
            Some((elements as f64) / (elapsed_ms * 1e6))
        } else {
            None
        };

        TimingResult {
            operation: operation.into(),
            elapsed_ms,
            throughput_gelem_s,
            elements,
        }
    }

    /// Print timing result in a human-readable format
    pub fn print(&self) {
        print!("{}: {:.3} ms", self.operation, self.elapsed_ms);
        if let Some(throughput) = self.throughput_gelem_s {
            print!(" ({:.2} Gelem/s)", throughput);
        }
        println!();
    }
}

/// Time a kernel operation and return the result with timing information
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::{khatri_rao, time_operation};
///
/// let a = Array2::<f64>::ones((100, 10));
/// let b = Array2::<f64>::ones((200, 10));
///
/// let (result, timing) = time_operation("khatri_rao", || {
///     khatri_rao(&a.view(), &b.view())
/// });
///
/// assert_eq!(result.shape(), &[20000, 10]);
/// assert!(timing.elapsed_ms > 0.0);
/// ```
pub fn time_operation<F, T>(name: impl Into<String>, op: F) -> (T, TimingResult)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = op();
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

    let timing = TimingResult {
        operation: name.into(),
        elapsed_ms,
        throughput_gelem_s: None,
        elements: 0,
    };

    (result, timing)
}

/// Compute the Frobenius norm of a matrix
///
/// The Frobenius norm is the square root of the sum of squared elements.
///
/// # Complexity
///
/// Time: O(rows × cols)
/// Space: O(1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::frobenius_norm;
///
/// let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let norm = frobenius_norm(&matrix.view());
/// assert!((norm - (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt()).abs() < 1e-10);
/// ```
pub fn frobenius_norm<T>(matrix: &ArrayView2<T>) -> T
where
    T: Clone + Num + Float,
{
    let mut sum = T::zero();
    for val in matrix.iter() {
        sum = sum + *val * *val;
    }
    sum.sqrt()
}

/// Compute the relative error between two matrices
///
/// Relative error = ||A - B||_F / ||A||_F
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::relative_error;
///
/// let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b = Array2::from_shape_vec((2, 2), vec![1.1, 2.1, 3.1, 4.1]).unwrap();
/// let error = relative_error(&a.view(), &b.view());
/// assert!(error > 0.0 && error < 0.1);
/// ```
pub fn relative_error<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> T
where
    T: Clone + Num + Float,
{
    if a.shape() != b.shape() {
        return T::infinity();
    }

    let diff = a.to_owned() - b.to_owned();
    let diff_norm = frobenius_norm(&diff.view());
    let a_norm = frobenius_norm(a);

    if a_norm > T::zero() {
        diff_norm / a_norm
    } else {
        diff_norm
    }
}

/// Check if two matrices are approximately equal within a tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::approx_equal;
///
/// let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b = Array2::from_shape_vec((2, 2), vec![1.0 + 1e-11, 2.0, 3.0, 4.0]).unwrap();
/// assert!(approx_equal(&a.view(), &b.view(), 1e-10));
/// ```
pub fn approx_equal<T>(a: &ArrayView2<T>, b: &ArrayView2<T>, tol: T) -> bool
where
    T: Clone + Num + Float,
{
    if a.shape() != b.shape() {
        return false;
    }

    for (a_val, b_val) in a.iter().zip(b.iter()) {
        if (*a_val - *b_val).abs() > tol {
            return false;
        }
    }

    true
}

/// Compute sparsity ratio of a matrix (fraction of zero elements)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::sparsity_ratio;
///
/// let matrix = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]).unwrap();
/// let sparsity = sparsity_ratio(&matrix.view(), 1e-10);
/// assert!((sparsity - 6.0 / 9.0).abs() < 1e-10);
/// ```
pub fn sparsity_ratio<T>(matrix: &ArrayView2<T>, zero_threshold: T) -> f64
where
    T: Clone + Num + Float,
{
    let total_elements = matrix.len();
    if total_elements == 0 {
        return 0.0;
    }

    let zero_count = matrix
        .iter()
        .filter(|&val| (*val).abs() <= zero_threshold)
        .count();

    zero_count as f64 / total_elements as f64
}

/// Generate a random matrix with specified shape and value range
///
/// Note: This uses a simple pseudo-random number generator.
/// For production use, consider using `scirs2_core::random` directly.
///
/// # Examples
///
/// ```
/// use tenrso_kernels::random_matrix;
///
/// let matrix = random_matrix::<f64>(10, 20, 0.0, 1.0);
/// assert_eq!(matrix.shape(), &[10, 20]);
/// for val in matrix.iter() {
///     assert!(*val >= 0.0 && *val <= 1.0);
/// }
/// ```
pub fn random_matrix<T>(rows: usize, cols: usize, min: T, max: T) -> Array2<T>
where
    T: Clone + Num + Float,
{
    use scirs2_core::ndarray_ext::Array2;

    Array2::from_shape_fn((rows, cols), |(i, j)| {
        // Simple deterministic pseudo-random (for reproducibility in tests)
        let seed = (i * 31 + j * 17) as f64;
        let normalized = (seed.sin().abs() * 1000.0).fract();
        let range = max - min;
        min + T::from(normalized).unwrap() * range
    })
}

/// Compute MTTKRP for all modes and return a vector of factor matrices
///
/// This is a convenience function that computes MTTKRP along each mode,
/// which is useful in CP-ALS iterations where all factors need updating.
///
/// # Arguments
///
/// * `tensor` - Input tensor view
/// * `factors` - Current factor matrices (one per mode)
///
/// # Returns
///
/// Vector of updated factor matrices, one per mode
///
/// # Complexity
///
/// Time: O(nmodes × MTTKRP_cost)
/// Space: O(nmodes × mode_size × rank)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_core::DenseND;
/// use tenrso_kernels::mttkrp_all_modes;
///
/// let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
/// let factors = vec![
///     Array2::<f64>::ones((3, 2)),
///     Array2::<f64>::ones((4, 2)),
///     Array2::<f64>::ones((5, 2)),
/// ];
/// let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
///
/// let updated = mttkrp_all_modes(&tensor.view(), &factor_views).unwrap();
/// assert_eq!(updated.len(), 3);
/// assert_eq!(updated[0].shape(), &[3, 2]);
/// assert_eq!(updated[1].shape(), &[4, 2]);
/// assert_eq!(updated[2].shape(), &[5, 2]);
/// ```
pub fn mttkrp_all_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
) -> KernelResult<Vec<Array2<T>>>
where
    T: Clone + Num + Float,
{
    let nmodes = tensor.ndim();

    if factors.len() != nmodes {
        return Err(KernelError::dimension_mismatch(
            "mttkrp_all_modes",
            vec![nmodes],
            vec![factors.len()],
            "Number of factors must match number of modes",
        ));
    }

    let mut result = Vec::with_capacity(nmodes);
    for mode in 0..nmodes {
        let factor_matrix = mttkrp(tensor, factors, mode)
            .map_err(|e| KernelError::operation_error("mttkrp_all_modes", e.to_string()))?;
        result.push(factor_matrix);
    }

    Ok(result)
}

/// Compute multiple Khatri-Rao products in batch
///
/// Given a list of matrix pairs, compute their Khatri-Rao products.
/// This is useful when you need to compute multiple Khatri-Rao products
/// in a decomposition workflow.
///
/// # Arguments
///
/// * `pairs` - Vector of (A, B) matrix pairs to compute KR products for
///
/// # Returns
///
/// Vector of Khatri-Rao products, one per input pair
///
/// # Complexity
///
/// Time: O(npairs × KR_cost)
/// Space: O(npairs × output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::khatri_rao_batch;
///
/// let a1 = Array2::<f64>::ones((10, 5));
/// let b1 = Array2::<f64>::ones((8, 5));
/// let a2 = Array2::<f64>::ones((6, 3));
/// let b2 = Array2::<f64>::ones((4, 3));
///
/// let pairs = vec![(a1.view(), b1.view()), (a2.view(), b2.view())];
/// let results = khatri_rao_batch(&pairs).unwrap();
///
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].shape(), &[80, 5]);
/// assert_eq!(results[1].shape(), &[24, 3]);
/// ```
pub fn khatri_rao_batch<T>(pairs: &[(ArrayView2<T>, ArrayView2<T>)]) -> KernelResult<Vec<Array2<T>>>
where
    T: Clone + Num + Float,
{
    if pairs.is_empty() {
        return Err(KernelError::empty_input("khatri_rao_batch", "pairs"));
    }

    let results = pairs.iter().map(|(a, b)| khatri_rao(a, b)).collect();

    Ok(results)
}

/// Normalize columns of a factor matrix to unit norm
///
/// Each column is divided by its L2 norm, making it a unit vector.
/// Returns the normalized matrix and a vector of the original norms.
///
/// This is commonly used in CP-ALS to prevent numerical instability.
///
/// # Arguments
///
/// * `factor` - Factor matrix to normalize (columns will be normalized)
///
/// # Returns
///
/// Tuple of (normalized_matrix, column_norms)
///
/// # Complexity
///
/// Time: O(rows × cols)
/// Space: O(rows × cols + cols)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::normalize_factor;
///
/// let factor = Array2::from_shape_vec((3, 2), vec![3.0_f64, 0.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
/// let (normalized, norms) = normalize_factor(&factor.view());
///
/// assert_eq!(norms.len(), 2);
/// assert!((norms[0] - 5.0_f64).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
/// assert!((norms[1] - 1.0_f64).abs() < 1e-10); // sqrt(1^2) = 1
///
/// // Check first column is normalized
/// let col0_norm: f64 = normalized.column(0).iter().map(|x| x * x).sum::<f64>().sqrt();
/// assert!((col0_norm - 1.0).abs() < 1e-10);
/// ```
pub fn normalize_factor<T>(factor: &ArrayView2<T>) -> (Array2<T>, Vec<T>)
where
    T: Clone + Num + Float,
{
    let (_nrows, ncols) = factor.dim();
    let mut normalized = factor.to_owned();
    let mut norms = Vec::with_capacity(ncols);

    for col_idx in 0..ncols {
        let mut col = normalized.column_mut(col_idx);

        // Compute L2 norm of column
        let mut norm_sq = T::zero();
        for val in col.iter() {
            norm_sq = norm_sq + *val * *val;
        }
        let norm = norm_sq.sqrt();
        norms.push(norm);

        // Normalize column (avoid division by zero)
        if norm > T::from(1e-15).unwrap() {
            for val in col.iter_mut() {
                *val = *val / norm;
            }
        }
    }

    (normalized, norms)
}

/// Denormalize a factor matrix using previously saved norms
///
/// Reverses the normalization done by `normalize_factor()` by
/// multiplying each column by its original norm.
///
/// # Arguments
///
/// * `normalized` - Normalized factor matrix
/// * `norms` - Column norms (from `normalize_factor()`)
///
/// # Returns
///
/// Denormalized factor matrix
///
/// # Complexity
///
/// Time: O(rows × cols)
/// Space: O(rows × cols)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::{normalize_factor, denormalize_factor};
///
/// let original = Array2::from_shape_vec((3, 2), vec![3.0_f64, 0.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
/// let (normalized, norms) = normalize_factor(&original.view());
/// let recovered = denormalize_factor(&normalized.view(), &norms);
///
/// // Should recover original matrix
/// for (a, b) in original.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-10_f64);
/// }
/// ```
pub fn denormalize_factor<T>(normalized: &ArrayView2<T>, norms: &[T]) -> Array2<T>
where
    T: Clone + Num + Float,
{
    let (_nrows, ncols) = normalized.dim();
    assert_eq!(
        ncols,
        norms.len(),
        "Number of norms must match number of columns"
    );

    let mut denormalized = normalized.to_owned();

    for (col_idx, &norm) in norms.iter().enumerate() {
        let mut col = denormalized.column_mut(col_idx);

        for val in col.iter_mut() {
            *val = *val * norm;
        }
    }

    denormalized
}

/// Validate that factor matrices have compatible shapes for CP decomposition
///
/// Checks that:
/// 1. All factors have the same number of columns (rank)
/// 2. Factor row counts match tensor dimensions
///
/// # Arguments
///
/// * `tensor_shape` - Shape of the tensor being decomposed
/// * `factors` - Factor matrices to validate
///
/// # Returns
///
/// `Ok(rank)` if valid, `Err` otherwise
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::validate_factor_shapes;
///
/// let factors = vec![
///     Array2::<f64>::ones((3, 5)),
///     Array2::<f64>::ones((4, 5)),
///     Array2::<f64>::ones((6, 5)),
/// ];
/// let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
///
/// let rank = validate_factor_shapes(&[3, 4, 6], &factor_views).unwrap();
/// assert_eq!(rank, 5);
///
/// // Mismatched rank should fail
/// let bad_factors = vec![
///     Array2::<f64>::ones((3, 5)),
///     Array2::<f64>::ones((4, 3)), // Wrong rank!
/// ];
/// let bad_views: Vec<_> = bad_factors.iter().map(|f| f.view()).collect();
/// assert!(validate_factor_shapes(&[3, 4], &bad_views).is_err());
/// ```
pub fn validate_factor_shapes<T>(
    tensor_shape: &[usize],
    factors: &[ArrayView2<T>],
) -> KernelResult<usize>
where
    T: Clone + Num,
{
    let nmodes = tensor_shape.len();

    // Check number of factors matches tensor order
    if factors.len() != nmodes {
        return Err(KernelError::dimension_mismatch(
            "validate_factor_shapes",
            vec![nmodes],
            vec![factors.len()],
            "Number of factors must match tensor order",
        ));
    }

    // Check factor row counts match tensor dimensions
    for (mode, factor) in factors.iter().enumerate() {
        let (nrows, _) = factor.dim();
        if nrows != tensor_shape[mode] {
            return Err(KernelError::dimension_mismatch(
                "validate_factor_shapes",
                vec![tensor_shape[mode]],
                vec![nrows],
                format!("Factor {} row count must match mode {} size", mode, mode),
            ));
        }
    }

    // Check all factors have same rank (number of columns)
    if factors.is_empty() {
        return Err(KernelError::empty_input(
            "validate_factor_shapes",
            "factors",
        ));
    }

    let rank = factors[0].ncols();
    for (i, factor) in factors.iter().enumerate().skip(1) {
        let factor_rank = factor.ncols();
        if factor_rank != rank {
            return Err(KernelError::rank_mismatch(
                "validate_factor_shapes",
                rank,
                factor_rank,
                i,
            ));
        }
    }

    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;
    use tenrso_core::DenseND;

    #[test]
    fn test_timing_result_new() {
        let result = TimingResult::new("test_op", 100.0, 1_000_000);
        assert_eq!(result.operation, "test_op");
        assert_eq!(result.elapsed_ms, 100.0);
        assert_eq!(result.elements, 1_000_000);
        assert!(result.throughput_gelem_s.is_some());
    }

    #[test]
    fn test_time_operation() {
        let (result, timing) = time_operation("add", || 2 + 2);
        assert_eq!(result, 4);
        assert!(timing.elapsed_ms >= 0.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let norm = frobenius_norm(&matrix.view());
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_error_identical() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let error = relative_error(&a.view(), &a.view());
        assert!(error < 1e-15);
    }

    #[test]
    fn test_relative_error_different_shapes() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let error = relative_error(&a.view(), &b.view());
        assert!(error.is_infinite());
    }

    #[test]
    fn test_approx_equal() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0 + 1e-11, 2.0, 3.0, 4.0]).unwrap();
        assert!(approx_equal(&a.view(), &b.view(), 1e-10));
        assert!(!approx_equal(&a.view(), &b.view(), 1e-12));
    }

    #[test]
    fn test_sparsity_ratio() {
        let sparse =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
                .unwrap();
        let sparsity = sparsity_ratio(&sparse.view(), 1e-10);
        assert!((sparsity - 6.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_ratio_dense() {
        let dense = Array2::ones((5, 5));
        let sparsity = sparsity_ratio(&dense.view(), 1e-10);
        assert!(sparsity.abs() < 1e-10);
    }

    #[test]
    fn test_random_matrix_shape() {
        let matrix = random_matrix::<f64>(10, 20, 0.0, 1.0);
        assert_eq!(matrix.shape(), &[10, 20]);
    }

    #[test]
    fn test_random_matrix_range() {
        let matrix = random_matrix::<f64>(5, 5, -1.0, 1.0);
        for val in matrix.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_mttkrp_all_modes() {
        let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
        let factors = [
            Array2::<f64>::ones((3, 2)),
            Array2::<f64>::ones((4, 2)),
            Array2::<f64>::ones((5, 2)),
        ];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let updated = mttkrp_all_modes(&tensor.view(), &factor_views).unwrap();

        assert_eq!(updated.len(), 3);
        assert_eq!(updated[0].shape(), &[3, 2]);
        assert_eq!(updated[1].shape(), &[4, 2]);
        assert_eq!(updated[2].shape(), &[5, 2]);
    }

    #[test]
    fn test_mttkrp_all_modes_wrong_num_factors() {
        let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
        let factors = [
            Array2::<f64>::ones((3, 2)),
            Array2::<f64>::ones((4, 2)),
            // Missing third factor!
        ];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result = mttkrp_all_modes(&tensor.view(), &factor_views);
        assert!(result.is_err());
    }

    #[test]
    fn test_khatri_rao_batch() {
        let a1 = Array2::<f64>::ones((10, 5));
        let b1 = Array2::<f64>::ones((8, 5));
        let a2 = Array2::<f64>::ones((6, 3));
        let b2 = Array2::<f64>::ones((4, 3));

        let pairs = vec![(a1.view(), b1.view()), (a2.view(), b2.view())];
        let results = khatri_rao_batch(&pairs).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].shape(), &[80, 5]);
        assert_eq!(results[1].shape(), &[24, 3]);
    }

    #[test]
    fn test_khatri_rao_batch_empty() {
        let pairs: Vec<(ArrayView2<f64>, ArrayView2<f64>)> = vec![];
        let result = khatri_rao_batch(&pairs);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_factor() {
        let factor = Array2::from_shape_vec((3, 2), vec![3.0, 0.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
        let (normalized, norms) = normalize_factor(&factor.view());

        assert_eq!(norms.len(), 2);
        assert!((norms[0] - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
        assert!((norms[1] - 1.0).abs() < 1e-10); // sqrt(1^2) = 1

        // Check first column is normalized
        let col0_norm: f64 = normalized
            .column(0)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!((col0_norm - 1.0).abs() < 1e-10);

        // Check second column is normalized
        let col1_norm: f64 = normalized
            .column(1)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!((col1_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_factor_zero_column() {
        let factor = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let (normalized, norms) = normalize_factor(&factor.view());

        assert_eq!(norms.len(), 2);
        assert!(norms[0] > 0.0);
        assert!((norms[1] - 0.0).abs() < 1e-15); // Zero column

        // Zero column should remain zero (not NaN)
        for val in normalized.column(1).iter() {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_denormalize_factor() {
        let original = Array2::from_shape_vec((3, 2), vec![3.0, 0.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
        let (normalized, norms) = normalize_factor(&original.view());
        let recovered = denormalize_factor(&normalized.view(), &norms);

        // Should recover original matrix
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let original = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let (normalized, norms) = normalize_factor(&original.view());
        let recovered = denormalize_factor(&normalized.view(), &norms);

        // Verify roundtrip
        assert!(approx_equal(&original.view(), &recovered.view(), 1e-10));
    }

    #[test]
    fn test_validate_factor_shapes() {
        let factors = [
            Array2::<f64>::ones((3, 5)),
            Array2::<f64>::ones((4, 5)),
            Array2::<f64>::ones((6, 5)),
        ];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let rank = validate_factor_shapes(&[3, 4, 6], &factor_views).unwrap();
        assert_eq!(rank, 5);
    }

    #[test]
    fn test_validate_factor_shapes_wrong_mode_size() {
        let factors = [
            Array2::<f64>::ones((3, 5)),
            Array2::<f64>::ones((4, 5)),
            Array2::<f64>::ones((7, 5)), // Should be 6!
        ];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result = validate_factor_shapes(&[3, 4, 6], &factor_views);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_factor_shapes_rank_mismatch() {
        let factors = [
            Array2::<f64>::ones((3, 5)),
            Array2::<f64>::ones((4, 3)), // Wrong rank!
        ];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result = validate_factor_shapes(&[3, 4], &factor_views);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_factor_shapes_wrong_num_factors() {
        let factors = [
            Array2::<f64>::ones((3, 5)),
            Array2::<f64>::ones((4, 5)),
            // Missing third factor!
        ];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result = validate_factor_shapes(&[3, 4, 6], &factor_views);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_factor_shapes_empty() {
        let factors: Vec<Array2<f64>> = vec![];
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result = validate_factor_shapes(&[], &factor_views);
        assert!(result.is_err());
    }
}
