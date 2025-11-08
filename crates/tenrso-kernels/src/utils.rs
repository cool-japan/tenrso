//! Utility functions and helpers for tensor kernel operations
//!
//! This module provides convenience functions and performance helpers
//! that simplify common patterns when working with tensor kernels.

use scirs2_core::ndarray_ext::{Array2, ArrayView2};
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;

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
}
