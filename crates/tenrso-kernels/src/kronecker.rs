//! Kronecker product implementation
//!
//! The Kronecker product is a matrix operation that constructs a block matrix
//! where each element of the first matrix is multiplied by the entire second matrix.
//! For matrices A (m×n) and B (p×q), the result C = A ⊗ B has size (mp×nq).
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::numeric::Num;

/// Compute the Kronecker product of two matrices
///
/// For matrices A (m×n) and B (p×q), the result C = A ⊗ B has size (mp×nq).
/// Each element a_ij of A is multiplied by the entire matrix B, creating a block matrix.
///
/// The resulting matrix has the structure:
/// ```text
/// [ a11*B  a12*B  ...  a1n*B ]
/// [ a21*B  a22*B  ...  a2n*B ]
/// [  ...    ...   ...   ...  ]
/// [ am1*B  am2*B  ...  amn*B ]
/// ```
///
/// # Arguments
///
/// * `a` - First matrix with shape (m, n)
/// * `b` - Second matrix with shape (p, q)
///
/// # Returns
///
/// A matrix with shape (mp, nq) containing the Kronecker product
///
/// # Complexity
///
/// Time: O(m * n * p * q)
/// Space: O(m * n * p * q)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::kronecker;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];  // 2×2
/// let b = array![[5.0, 6.0], [7.0, 8.0]];  // 2×2
/// let c = kronecker(&a.view(), &b.view());  // 4×4
/// assert_eq!(c.shape(), &[4, 4]);
///
/// // First block (top-left): 1*B
/// assert_eq!(c[[0, 0]], 5.0);   // 1*5
/// assert_eq!(c[[0, 1]], 6.0);   // 1*6
/// assert_eq!(c[[1, 0]], 7.0);   // 1*7
/// assert_eq!(c[[1, 1]], 8.0);   // 1*8
/// ```
pub fn kronecker<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Num,
{
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let (p, q) = (b.shape()[0], b.shape()[1]);

    // Allocate output matrix (mp × nq)
    let mut result = Array2::<T>::zeros((m * p, n * q));

    // For each element in A, multiply by the entire B matrix
    for (i, row_a) in a.rows().into_iter().enumerate() {
        for (j, a_val) in row_a.iter().enumerate() {
            // Compute the block starting position
            let block_row = i * p;
            let block_col = j * q;

            // Fill in the block: a_ij * B
            for (bi, row_b) in b.rows().into_iter().enumerate() {
                for (bj, b_val) in row_b.iter().enumerate() {
                    result[[block_row + bi, block_col + bj]] = a_val.clone() * b_val.clone();
                }
            }
        }
    }

    result
}

/// Compute the Kronecker product with parallel execution
///
/// This is a parallel version of `kronecker` that processes rows in parallel
/// using Rayon. Use this for large matrices where the overhead of parallelization
/// is justified.
///
/// # Arguments
///
/// * `a` - First matrix with shape (m, n)
/// * `b` - Second matrix with shape (p, q)
///
/// # Returns
///
/// A matrix with shape (mp, nq) containing the Kronecker product
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::kronecker_parallel;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let c = kronecker_parallel(&a.view(), &b.view());
/// assert_eq!(c.shape(), &[4, 4]);
/// ```
#[cfg(feature = "parallel")]
pub fn kronecker_parallel<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Num + Send + Sync,
{
    use scirs2_core::ndarray_ext::Axis;
    use scirs2_core::parallel_ops::*;

    let (m, n) = (a.shape()[0], a.shape()[1]);
    let (p, q) = (b.shape()[0], b.shape()[1]);

    // Allocate output matrix (mp × nq)
    let mut result = Array2::<T>::zeros((m * p, n * q));

    // Process each row of A in parallel
    result
        .axis_chunks_iter_mut(Axis(0), p)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut result_block)| {
            // For each element in this row of A
            for j in 0..n {
                let a_val = &a[[i, j]];
                let block_col = j * q;

                // Fill in the block: a_ij * B
                for bi in 0..p {
                    for bj in 0..q {
                        let b_val = &b[[bi, bj]];
                        result_block[[bi, block_col + bj]] = a_val.clone() * b_val.clone();
                    }
                }
            }
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_kronecker_basic() {
        // Simple 2×2 example
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = kronecker(&a.view(), &b.view());

        assert_eq!(c.shape(), &[4, 4]);

        // First block (top-left): 1*B = B
        assert_eq!(c[[0, 0]], 5.0);
        assert_eq!(c[[0, 1]], 6.0);
        assert_eq!(c[[1, 0]], 7.0);
        assert_eq!(c[[1, 1]], 8.0);

        // Second block (top-right): 2*B
        assert_eq!(c[[0, 2]], 10.0); // 2*5
        assert_eq!(c[[0, 3]], 12.0); // 2*6
        assert_eq!(c[[1, 2]], 14.0); // 2*7
        assert_eq!(c[[1, 3]], 16.0); // 2*8

        // Third block (bottom-left): 3*B
        assert_eq!(c[[2, 0]], 15.0); // 3*5
        assert_eq!(c[[2, 1]], 18.0); // 3*6
        assert_eq!(c[[3, 0]], 21.0); // 3*7
        assert_eq!(c[[3, 1]], 24.0); // 3*8

        // Fourth block (bottom-right): 4*B
        assert_eq!(c[[2, 2]], 20.0); // 4*5
        assert_eq!(c[[2, 3]], 24.0); // 4*6
        assert_eq!(c[[3, 2]], 28.0); // 4*7
        assert_eq!(c[[3, 3]], 32.0); // 4*8
    }

    #[test]
    fn test_kronecker_different_sizes() {
        // A: 2×3, B: 2×2 => Result: 4×6
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0]];
        let c = kronecker(&a.view(), &b.view());

        assert_eq!(c.shape(), &[4, 6]);

        // First block: 1*B
        assert_eq!(c[[0, 0]], 7.0);
        assert_eq!(c[[0, 1]], 8.0);
        assert_eq!(c[[1, 0]], 9.0);
        assert_eq!(c[[1, 1]], 10.0);

        // Last block: 6*B
        assert_eq!(c[[2, 4]], 42.0); // 6*7
        assert_eq!(c[[2, 5]], 48.0); // 6*8
        assert_eq!(c[[3, 4]], 54.0); // 6*9
        assert_eq!(c[[3, 5]], 60.0); // 6*10
    }

    #[test]
    fn test_kronecker_identity() {
        // Test with identity matrix
        let i2 = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[2.0, 3.0], [4.0, 5.0]];
        let c = kronecker(&i2.view(), &b.view());

        assert_eq!(c.shape(), &[4, 4]);

        // I2 ⊗ B should give a block diagonal matrix
        // Top-left block: 1*B = B
        assert_eq!(c[[0, 0]], 2.0);
        assert_eq!(c[[0, 1]], 3.0);
        assert_eq!(c[[1, 0]], 4.0);
        assert_eq!(c[[1, 1]], 5.0);

        // Top-right block: 0*B = zeros
        assert_eq!(c[[0, 2]], 0.0);
        assert_eq!(c[[0, 3]], 0.0);
        assert_eq!(c[[1, 2]], 0.0);
        assert_eq!(c[[1, 3]], 0.0);

        // Bottom-left block: 0*B = zeros
        assert_eq!(c[[2, 0]], 0.0);
        assert_eq!(c[[2, 1]], 0.0);
        assert_eq!(c[[3, 0]], 0.0);
        assert_eq!(c[[3, 1]], 0.0);

        // Bottom-right block: 1*B = B
        assert_eq!(c[[2, 2]], 2.0);
        assert_eq!(c[[2, 3]], 3.0);
        assert_eq!(c[[3, 2]], 4.0);
        assert_eq!(c[[3, 3]], 5.0);
    }

    #[test]
    fn test_kronecker_single_element() {
        // Test with 1×1 matrices (scalars)
        let a = array![[3.0]];
        let b = array![[5.0]];
        let c = kronecker(&a.view(), &b.view());

        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c[[0, 0]], 15.0); // 3*5
    }

    #[test]
    fn test_kronecker_vector() {
        // Test with column vectors (nx1 matrices)
        let a = array![[2.0], [3.0]];
        let b = array![[4.0], [5.0]];
        let c = kronecker(&a.view(), &b.view());

        assert_eq!(c.shape(), &[4, 1]);
        assert_eq!(c[[0, 0]], 8.0); // 2*4
        assert_eq!(c[[1, 0]], 10.0); // 2*5
        assert_eq!(c[[2, 0]], 12.0); // 3*4
        assert_eq!(c[[3, 0]], 15.0); // 3*5
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_kronecker_parallel() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let c_serial = kronecker(&a.view(), &b.view());
        let c_parallel = kronecker_parallel(&a.view(), &b.view());

        assert_eq!(c_serial.shape(), c_parallel.shape());

        // Check all elements match
        for i in 0..c_serial.shape()[0] {
            for j in 0..c_serial.shape()[1] {
                assert_eq!(c_serial[[i, j]], c_parallel[[i, j]]);
            }
        }
    }
}
