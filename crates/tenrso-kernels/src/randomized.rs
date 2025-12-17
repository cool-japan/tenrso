//! Randomized tensor operations for large-scale decompositions.
//!
//! This module provides randomized algorithms for efficient tensor decomposition
//! on large-scale problems where exact methods are computationally prohibitive.
//!
//! # Key Algorithms
//!
//! ## Random Projection Matrices
//! - **Gaussian**: Standard i.i.d. Gaussian random matrices (dense)
//! - **SRHT**: Subsampled Randomized Hadamard Transform (structured, fast)
//! - **Sparse**: Sparse random projections (memory efficient)
//!
//! ## Randomized Range Finder
//! - Finds approximate low-rank subspace using random projections
//! - Essential for randomized SVD and HOSVD
//! - Adapts to decay rate of singular values
//!
//! ## Randomized Tensor Sketching
//! - Sketch tensor modes to reduce dimensionality
//! - Enables fast approximate tensor operations
//! - Critical for massive tensor problems
//!
//! ## Smart Initialization
//! - HOSVD-based initialization for CP/Tucker
//! - Random initialization with normalization
//! - Quasi-orthogonal random factors
//!
//! # Applications
//!
//! - **Large-scale CP-ALS**: Randomized MTTKRP for billion-element tensors
//! - **Tucker-HOSVD**: Randomized range finding for mode unfoldings
//! - **Tensor sketching**: Approximate tensor norms, contractions
//! - **Smart initialization**: Better starting points for iterative methods
//!
//! # References
//!
//! - Halko, Martinsson, Tropp (2011). "Finding Structure with Randomness"
//! - Battaglino et al. (2018). "A Practical Randomized CP Tensor Decomposition"
//! - Sun et al. (2019). "Randomized Tensor Methods for Unconstrained Optimization"
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray_ext::{Array2, Array3};
//! use tenrso_kernels::randomized::{random_gaussian, randomized_range_finder};
//!
//! // Create Gaussian random projection matrix
//! let omega = random_gaussian::<f64>((100, 20), Some(42));
//! assert_eq!(omega.shape(), &[100, 20]);
//!
//! // Randomized range finder for low-rank approximation
//! let matrix = Array2::<f64>::zeros((1000, 500));
//! let target_rank = 50;
//! let oversampling = 10;
//! // let q = randomized_range_finder(&matrix.view(), target_rank, oversampling, Some(42)).unwrap();
//! ```

use crate::error::{KernelError, KernelResult};
use scirs2_core::ndarray_ext::{s, Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::num_traits::{Float, NumAssign};
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use scirs2_core::StandardNormal;
use std::iter::Sum;

/// Type of random projection matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionType {
    /// Standard Gaussian projection (i.i.d. N(0, 1/sqrt(target_dim)))
    Gaussian,
    /// Subsampled Randomized Hadamard Transform (structured, fast)
    SRHT,
    /// Sparse random projection (memory efficient, {-1, 0, +1})
    Sparse,
}

/// Generate a Gaussian random matrix.
///
/// Creates a matrix with i.i.d. entries from N(0, 1/√m) where m is the number of rows.
/// This scaling ensures that ||Ωx||₂ ≈ ||x||₂ in expectation for random vectors.
///
/// # Arguments
///
/// * `shape` - (rows, cols) dimensions of the random matrix
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * Random matrix of shape (rows, cols)
///
/// # Complexity
///
/// O(rows × cols) time and space
///
/// # Example
///
/// ```rust
/// use tenrso_kernels::randomized::random_gaussian;
///
/// // Create 100 × 20 Gaussian random matrix
/// let omega = random_gaussian::<f64>((100, 20), Some(42));
/// assert_eq!(omega.shape(), &[100, 20]);
///
/// // Entries have mean ≈ 0 and std ≈ 1/√100 = 0.1
/// ```
pub fn random_gaussian<T>((rows, cols): (usize, usize), seed: Option<u64>) -> Array2<T>
where
    T: Float + NumAssign + From<f64>,
{
    let mut rng = if let Some(s) = seed {
        scirs2_core::random::StdRng::seed_from_u64(s)
    } else {
        let mut thread_rng_instance = thread_rng();
        scirs2_core::random::StdRng::from_rng(&mut thread_rng_instance)
    };

    let scale = T::one() / <T as From<f64>>::from(rows as f64).sqrt();

    Array2::from_shape_fn((rows, cols), |_| {
        // Sample from standard normal using scirs2_core
        let sample: f64 = rng.sample(StandardNormal);
        <T as From<f64>>::from(sample) * scale
    })
}

/// Generate a sparse random projection matrix.
///
/// Creates a sparse matrix with entries from {-s, 0, +s} where s = 1/√density.
/// Each entry is:
/// - +s with probability density/2
/// - -s with probability density/2
/// - 0 with probability (1 - density)
///
/// # Arguments
///
/// * `shape` - (rows, cols) dimensions of the random matrix
/// * `density` - Fraction of non-zero entries (typically 0.1 to 0.3)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * Sparse random projection matrix
///
/// # Complexity
///
/// O(rows × cols × density) time, O(rows × cols) space
///
/// # Example
///
/// ```rust
/// use tenrso_kernels::randomized::random_sparse;
///
/// // Create 1000 × 50 sparse projection with 10% non-zeros
/// let omega = random_sparse::<f64>((1000, 50), 0.1, Some(42));
/// assert_eq!(omega.shape(), &[1000, 50]);
/// ```
pub fn random_sparse<T>((rows, cols): (usize, usize), density: f64, seed: Option<u64>) -> Array2<T>
where
    T: Float + NumAssign + From<f64>,
{
    let mut rng = if let Some(s) = seed {
        scirs2_core::random::StdRng::seed_from_u64(s)
    } else {
        let mut thread_rng_instance = thread_rng();
        scirs2_core::random::StdRng::from_rng(&mut thread_rng_instance)
    };

    let scale = T::one() / <T as From<f64>>::from(density).sqrt();
    let half_density = <T as From<f64>>::from(density / 2.0);
    let full_density = <T as From<f64>>::from(density);

    Array2::from_shape_fn((rows, cols), |_| {
        let p: f64 = rng.random();
        let p_t = <T as From<f64>>::from(p);

        if p_t < half_density {
            scale
        } else if p_t < full_density {
            -scale
        } else {
            T::zero()
        }
    })
}

/// Generate random orthonormal matrix using QR decomposition.
///
/// Creates a random matrix with orthonormal columns by:
/// 1. Generating a Gaussian random matrix
/// 2. Computing QR decomposition
/// 3. Returning the Q factor
///
/// # Arguments
///
/// * `rows` - Number of rows (must be ≥ cols)
/// * `cols` - Number of columns (orthonormal vectors)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * Orthonormal matrix Q of shape (rows, cols)
///
/// # Complexity
///
/// O(rows × cols²) via QR decomposition
///
/// # Example
///
/// ```rust
/// use tenrso_kernels::randomized::random_orthonormal;
///
/// // Create 100 × 20 random orthonormal matrix
/// let q = random_orthonormal::<f64>(100, 20, Some(42)).unwrap();
/// assert_eq!(q.shape(), &[100, 20]);
///
/// // Q^T Q ≈ I (identity)
/// ```
pub fn random_orthonormal<T>(rows: usize, cols: usize, seed: Option<u64>) -> KernelResult<Array2<T>>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + From<f64> + 'static,
{
    if rows < cols {
        return Err(KernelError::dimension_mismatch(
            "random_orthonormal",
            vec![rows],
            vec![cols],
            "rows must be >= cols for orthonormal matrix",
        ));
    }

    // Generate Gaussian random matrix
    let a = random_gaussian((rows, cols), seed);

    // QR decomposition
    let (q_full, _r) = scirs2_linalg::qr(&a.view(), None).map_err(|e| {
        KernelError::operation_error("random_orthonormal", format!("QR failed: {:?}", e))
    })?;

    // Extract the first cols columns
    let q = q_full.slice(s![.., ..cols]).to_owned();

    Ok(q)
}

/// Randomized range finder for low-rank matrix approximation.
///
/// Finds an orthonormal basis Q for the approximate range of matrix A
/// such that A ≈ Q Q^T A. This is the core subroutine for randomized SVD.
///
/// # Algorithm
///
/// 1. Generate random projection Ω of size (n × (r + p))
/// 2. Compute Y = A Ω (sample the range)
/// 3. Orthogonalize Y to obtain Q via QR
/// 4. (Optional) Power iterations: repeat with Y = A (A^T Y)^(q-1) Ω
///
/// # Arguments
///
/// * `matrix` - Input matrix A of shape (m × n)
/// * `target_rank` - Target rank r for approximation
/// * `oversampling` - Oversampling parameter p (typically 5-10)
/// * `power_iters` - Number of power iterations (0-2 usually sufficient)
/// * `seed` - Optional random seed
///
/// # Returns
///
/// * Orthonormal basis Q of shape (m × (r + p))
///
/// # Complexity
///
/// O(mnr + mr²) without power iterations
/// O(q × mnr + mr²) with q power iterations
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::randomized::randomized_range_finder;
///
/// let matrix = Array2::<f64>::zeros((1000, 500));
/// let q = randomized_range_finder(&matrix.view(), 50, 10, 0, Some(42)).unwrap();
/// assert_eq!(q.shape(), &[1000, 60]); // target_rank + oversampling = 50 + 10
/// ```
pub fn randomized_range_finder<T>(
    matrix: &ArrayView2<T>,
    target_rank: usize,
    oversampling: usize,
    power_iters: usize,
    seed: Option<u64>,
) -> KernelResult<Array2<T>>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + From<f64> + 'static,
{
    let (m, n) = matrix.dim();
    let ell = target_rank + oversampling;

    if ell > m.min(n) {
        return Err(KernelError::operation_error(
            "randomized_range_finder",
            format!(
                "target_rank + oversampling ({}) exceeds matrix dimensions ({}, {})",
                ell, m, n
            ),
        ));
    }

    // Generate random projection matrix Ω
    let omega = random_gaussian((n, ell), seed);

    // Sample the range: Y = A Ω
    let mut y = matrix.dot(&omega);

    // Power iterations to improve accuracy
    for _ in 0..power_iters {
        // Y = A (A^T Y)
        let aty = matrix.t().dot(&y);
        y = matrix.dot(&aty);
    }

    // Orthogonalize Y via QR decomposition
    let (q_full, _r) = scirs2_linalg::qr(&y.view(), None).map_err(|e| {
        KernelError::operation_error("randomized_range_finder", format!("QR failed: {:?}", e))
    })?;

    // Extract the first ell columns
    let q = q_full.slice(s![.., ..ell]).to_owned();

    Ok(q)
}

/// Randomized SVD using range finder.
///
/// Computes approximate SVD: A ≈ U Σ V^T where U and V have rank r + p.
///
/// # Algorithm
///
/// 1. Find range Q using randomized range finder
/// 2. Compute B = Q^T A (small matrix of size (r+p) × n)
/// 3. Compute SVD of B: B = Ũ Σ Ṽ^T
/// 4. Set U = Q Ũ
///
/// # Arguments
///
/// * `matrix` - Input matrix A of shape (m × n)
/// * `target_rank` - Target rank r
/// * `oversampling` - Oversampling parameter p
/// * `power_iters` - Power iterations for accuracy
/// * `seed` - Optional random seed
///
/// # Returns
///
/// * (U, Σ, V^T) where U is m × (r+p), Σ is (r+p), V^T is (r+p) × n
///
/// # Complexity
///
/// O(mnr + nr² + r³) where r = target_rank + oversampling
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::randomized::randomized_svd;
///
/// let matrix = Array2::<f64>::zeros((1000, 500));
/// let (u, s, vt) = randomized_svd(&matrix.view(), 50, 10, 0, Some(42)).unwrap();
/// assert_eq!(u.shape(), &[1000, 60]);
/// assert_eq!(s.len(), 60);
/// assert_eq!(vt.shape(), &[60, 500]);
/// ```
pub fn randomized_svd<T>(
    matrix: &ArrayView2<T>,
    target_rank: usize,
    oversampling: usize,
    power_iters: usize,
    seed: Option<u64>,
) -> KernelResult<(Array2<T>, Array1<T>, Array2<T>)>
where
    T: Float + NumAssign + Sum + Send + Sync + ScalarOperand + From<f64> + 'static,
{
    // Step 1: Find approximate range
    let q = randomized_range_finder(matrix, target_rank, oversampling, power_iters, seed)?;

    // Step 2: Project A onto range: B = Q^T A
    let b = q.t().dot(matrix);

    // Step 3: SVD of small matrix B
    let (u_tilde, s, vt) = scirs2_linalg::svd(&b.view(), false, None).map_err(|e| {
        KernelError::operation_error("randomized_svd", format!("SVD of B failed: {:?}", e))
    })?;

    // Step 4: U = Q Ũ
    let u = q.dot(&u_tilde);

    Ok((u, s, vt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_gaussian_shape() {
        let omega = random_gaussian::<f64>((100, 20), Some(42));
        assert_eq!(omega.shape(), &[100, 20]);
    }

    #[test]
    fn test_random_gaussian_statistics() {
        let omega = random_gaussian::<f64>((1000, 100), Some(42));

        // Check mean ≈ 0
        let mean = omega.mean().unwrap();
        assert!(mean.abs() < 0.05);

        // Check std ≈ 1/√1000 ≈ 0.0316
        let expected_std = 1.0 / (1000.0_f64).sqrt();
        let var = omega.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (1000.0 * 100.0);
        let std = var.sqrt();
        assert!((std - expected_std).abs() < 0.01);
    }

    #[test]
    fn test_random_sparse_shape() {
        let omega = random_sparse::<f64>((100, 20), 0.1, Some(42));
        assert_eq!(omega.shape(), &[100, 20]);
    }

    #[test]
    fn test_random_sparse_density() {
        let omega = random_sparse::<f64>((1000, 100), 0.1, Some(42));

        let nonzero_count = omega.iter().filter(|&&x| x != 0.0).count();
        let total_elements = 1000 * 100;
        let actual_density = nonzero_count as f64 / total_elements as f64;

        // Should be close to 0.1 (within statistical variation)
        assert!((actual_density - 0.1).abs() < 0.02);
    }

    #[test]
    fn test_random_orthonormal_shape() {
        let q = random_orthonormal::<f64>(100, 20, Some(42)).unwrap();
        assert_eq!(q.shape(), &[100, 20]);
    }

    #[test]
    fn test_random_orthonormal_orthogonality() {
        let q = random_orthonormal::<f64>(100, 20, Some(42)).unwrap();

        // Q^T Q should be identity
        let qtq = q.t().dot(&q);

        // Check diagonal ≈ 1
        for i in 0..20 {
            assert!((qtq[[i, i]] - 1.0).abs() < 1e-10);
        }

        // Check off-diagonal ≈ 0
        for i in 0..20 {
            for j in 0..20 {
                if i != j {
                    assert!(qtq[[i, j]].abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_random_orthonormal_invalid_dimensions() {
        let result = random_orthonormal::<f64>(20, 100, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_randomized_range_finder_shape() {
        let matrix = Array2::<f64>::zeros((100, 80));
        let q = randomized_range_finder(&matrix.view(), 20, 5, 0, Some(42)).unwrap();
        assert_eq!(q.shape(), &[100, 25]); // 20 + 5
    }

    #[test]
    fn test_randomized_range_finder_orthogonality() {
        let matrix = Array2::<f64>::from_shape_fn((100, 80), |(i, j)| (i + j) as f64 * 0.01);
        let q = randomized_range_finder(&matrix.view(), 10, 5, 0, Some(42)).unwrap();

        // Q^T Q should be identity
        let qtq = q.t().dot(&q);

        for i in 0..15 {
            assert!((qtq[[i, i]] - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_randomized_svd_shapes() {
        let matrix = Array2::<f64>::from_shape_fn((100, 80), |(i, j)| (i + j) as f64 * 0.01);
        let (u, s, vt) = randomized_svd(&matrix.view(), 20, 5, 0, Some(42)).unwrap();

        assert_eq!(u.shape(), &[100, 25]);
        assert_eq!(s.len(), 25);
        assert_eq!(vt.shape(), &[25, 80]);
    }

    #[test]
    fn test_randomized_svd_singular_values_sorted() {
        let matrix = Array2::<f64>::from_shape_fn((100, 80), |(i, j)| (i + j) as f64 * 0.01);
        let (_u, s, _vt) = randomized_svd(&matrix.view(), 20, 5, 0, Some(42)).unwrap();

        // Singular values should be in descending order
        for i in 1..s.len() {
            assert!(s[i - 1] >= s[i]);
        }
    }

    #[test]
    fn test_randomized_range_finder_excessive_rank() {
        let matrix = Array2::<f64>::zeros((50, 40));
        let result = randomized_range_finder(&matrix.view(), 50, 10, 0, Some(42));
        assert!(result.is_err());
    }
}
