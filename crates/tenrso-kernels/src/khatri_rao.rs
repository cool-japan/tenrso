//! Khatri-Rao product (column-wise Kronecker product)
//!
//! The Khatri-Rao product is a fundamental operation in tensor decompositions,
//! especially CP-ALS. For matrices A (I × K) and B (J × K), the Khatri-Rao
//! product C = A ⊙ B has size (I*J × K) where each column k of C is the
//! Kronecker product of column k of A and column k of B.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use scirs2_core::ndarray_ext::{Array2, ArrayView2, Axis};
use scirs2_core::numeric::Num;

/// Compute the Khatri-Rao product (column-wise Kronecker product) of two matrices
///
/// For matrices A (I × K) and B (J × K), the result C = A ⊙ B has size (I*J × K).
/// Each column k of C is the Kronecker product of column k of A and column k of B.
///
/// # Arguments
///
/// * `a` - First matrix with shape (I, K)
/// * `b` - Second matrix with shape (J, K)
///
/// # Returns
///
/// A matrix with shape (I*J, K) containing the Khatri-Rao product
///
/// # Panics
///
/// Panics if the number of columns in A and B don't match
///
/// # Complexity
///
/// Time: O(I * J * K)
/// Space: O(I * J * K)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::khatri_rao;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];  // 2×2
/// let b = array![[5.0, 6.0], [7.0, 8.0]];  // 2×2
/// let c = khatri_rao(&a.view(), &b.view());  // 4×2
/// assert_eq!(c.shape(), &[4, 2]);
///
/// // First column: [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
/// assert_eq!(c[[0, 0]], 5.0);
/// assert_eq!(c[[1, 0]], 7.0);
/// assert_eq!(c[[2, 0]], 15.0);
/// assert_eq!(c[[3, 0]], 21.0);
/// ```
pub fn khatri_rao<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Num,
{
    let (i, k1) = (a.shape()[0], a.shape()[1]);
    let (j, k2) = (b.shape()[0], b.shape()[1]);

    assert_eq!(
        k1, k2,
        "Number of columns must match: A has {} columns, B has {} columns",
        k1, k2
    );

    let k = k1;

    // Allocate output matrix (I*J × K)
    let mut result = Array2::<T>::zeros((i * j, k));

    // For each column
    for col_idx in 0..k {
        let a_col = a.column(col_idx);
        let b_col = b.column(col_idx);

        // Compute Kronecker product of the two columns
        for (row_a_idx, a_val) in a_col.iter().enumerate() {
            for (row_b_idx, b_val) in b_col.iter().enumerate() {
                let result_row = row_a_idx * j + row_b_idx;
                result[[result_row, col_idx]] = a_val.clone() * b_val.clone();
            }
        }
    }

    result
}

/// Compute the Khatri-Rao product with parallel execution
///
/// This is a parallel version of `khatri_rao` that processes columns in parallel
/// using Rayon. Use this for large matrices where the overhead of parallelization
/// is justified.
///
/// # Arguments
///
/// * `a` - First matrix with shape (I, K)
/// * `b` - Second matrix with shape (J, K)
///
/// # Returns
///
/// A matrix with shape (I*J, K) containing the Khatri-Rao product
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::khatri_rao_parallel;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let c = khatri_rao_parallel(&a.view(), &b.view());
/// assert_eq!(c.shape(), &[4, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn khatri_rao_parallel<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Num + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let (i, k1) = (a.shape()[0], a.shape()[1]);
    let (j, k2) = (b.shape()[0], b.shape()[1]);

    assert_eq!(
        k1, k2,
        "Number of columns must match: A has {} columns, B has {} columns",
        k1, k2
    );

    let k = k1;

    // Allocate output matrix (I*J × K)
    let mut result = Array2::<T>::zeros((i * j, k));

    // Process columns in parallel
    result
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(col_idx, mut result_col)| {
            let a_col = a.column(col_idx);
            let b_col = b.column(col_idx);

            // Compute Kronecker product of the two columns
            for (row_a_idx, a_val) in a_col.iter().enumerate() {
                for (row_b_idx, b_val) in b_col.iter().enumerate() {
                    let result_row = row_a_idx * j + row_b_idx;
                    result_col[result_row] = a_val.clone() * b_val.clone();
                }
            }
        });

    result
}

/// Cache-blocked Khatri-Rao product for better cache locality
///
/// This version processes the matrices in tiles/blocks that fit better in cache,
/// reducing cache misses and improving performance for large matrices.
///
/// The algorithm divides both the row dimensions and column dimensions into blocks,
/// and processes each block independently. This improves spatial and temporal locality.
///
/// # Arguments
///
/// * `a` - First matrix with shape (I, K)
/// * `b` - Second matrix with shape (J, K)
/// * `block_size` - Size of blocks for tiling (in rows per matrix)
///
/// # Returns
///
/// A matrix with shape (I*J, K) containing the Khatri-Rao product
///
/// # Complexity
///
/// Same asymptotic complexity as standard Khatri-Rao: O(I * J * K)
/// But with better cache behavior for large matrices
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::khatri_rao_blocked;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];  // 4×2
/// let b = array![[9.0, 10.0], [11.0, 12.0]];  // 2×2
/// let c = khatri_rao_blocked(&a.view(), &b.view(), 2);  // Block size 2
/// assert_eq!(c.shape(), &[8, 2]);
/// ```
pub fn khatri_rao_blocked<T>(a: &ArrayView2<T>, b: &ArrayView2<T>, block_size: usize) -> Array2<T>
where
    T: Clone + Num,
{
    let (i, k1) = (a.shape()[0], a.shape()[1]);
    let (j, k2) = (b.shape()[0], b.shape()[1]);

    assert_eq!(
        k1, k2,
        "Number of columns must match: A has {} columns, B has {} columns",
        k1, k2
    );

    assert!(
        block_size > 0,
        "Block size must be greater than 0, got {}",
        block_size
    );

    let k = k1;

    // For small matrices, use standard implementation
    if i * j < block_size * block_size * 4 {
        return khatri_rao(a, b);
    }

    // Allocate output matrix (I*J × K)
    let mut result = Array2::<T>::zeros((i * j, k));

    // Compute number of blocks
    let num_blocks_a = i.div_ceil(block_size);
    let num_blocks_b = j.div_ceil(block_size);

    // Process in blocks
    for block_a_idx in 0..num_blocks_a {
        let row_a_start = block_a_idx * block_size;
        let row_a_end = (row_a_start + block_size).min(i);

        for block_b_idx in 0..num_blocks_b {
            let row_b_start = block_b_idx * block_size;
            let row_b_end = (row_b_start + block_size).min(j);

            // Process this block for all columns
            for col_idx in 0..k {
                let a_col = a.column(col_idx);
                let b_col = b.column(col_idx);

                // Compute Kronecker product for this block
                for row_a_idx in row_a_start..row_a_end {
                    let a_val = &a_col[row_a_idx];

                    for row_b_idx in row_b_start..row_b_end {
                        let b_val = &b_col[row_b_idx];
                        let result_row = row_a_idx * j + row_b_idx;
                        result[[result_row, col_idx]] = a_val.clone() * b_val.clone();
                    }
                }
            }
        }
    }

    result
}

/// Parallel cache-blocked Khatri-Rao product
///
/// Combines cache-blocking with parallel execution for maximum performance
/// on large matrices with multi-core systems.
///
/// # Arguments
///
/// * `a` - First matrix with shape (I, K)
/// * `b` - Second matrix with shape (J, K)
/// * `block_size` - Size of blocks for tiling
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::khatri_rao_blocked_parallel;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let b = array![[9.0, 10.0], [11.0, 12.0]];
/// let c = khatri_rao_blocked_parallel(&a.view(), &b.view(), 2);
/// assert_eq!(c.shape(), &[8, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn khatri_rao_blocked_parallel<T>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    block_size: usize,
) -> Array2<T>
where
    T: Clone + Num + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let (i, k1) = (a.shape()[0], a.shape()[1]);
    let (j, k2) = (b.shape()[0], b.shape()[1]);

    assert_eq!(
        k1, k2,
        "Number of columns must match: A has {} columns, B has {} columns",
        k1, k2
    );

    assert!(
        block_size > 0,
        "Block size must be greater than 0, got {}",
        block_size
    );

    let k = k1;

    // For small matrices, use standard parallel implementation
    if i * j < block_size * block_size * 4 {
        return khatri_rao_parallel(a, b);
    }

    // Allocate output matrix (I*J × K)
    let mut result = Array2::<T>::zeros((i * j, k));

    // Compute number of blocks
    let num_blocks_a = i.div_ceil(block_size);
    let num_blocks_b = j.div_ceil(block_size);

    // Create block pairs for parallel processing
    let mut block_pairs = Vec::new();
    for block_a_idx in 0..num_blocks_a {
        for block_b_idx in 0..num_blocks_b {
            block_pairs.push((block_a_idx, block_b_idx));
        }
    }

    // Process column blocks in parallel
    result
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(col_idx, mut result_col)| {
            let a_col = a.column(col_idx);
            let b_col = b.column(col_idx);

            // Process each block pair
            for &(block_a_idx, block_b_idx) in &block_pairs {
                let row_a_start = block_a_idx * block_size;
                let row_a_end = (row_a_start + block_size).min(i);
                let row_b_start = block_b_idx * block_size;
                let row_b_end = (row_b_start + block_size).min(j);

                // Compute Kronecker product for this block
                for row_a_idx in row_a_start..row_a_end {
                    let a_val = &a_col[row_a_idx];

                    for row_b_idx in row_b_start..row_b_end {
                        let b_val = &b_col[row_b_idx];
                        let result_row = row_a_idx * j + row_b_idx;
                        result_col[result_row] = a_val.clone() * b_val.clone();
                    }
                }
            }
        });

    result
}

#[cfg(test)]
mod blocked_tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_khatri_rao_blocked_matches_standard() {
        let a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let c_standard = khatri_rao(&a.view(), &b.view());
        let c_blocked = khatri_rao_blocked(&a.view(), &b.view(), 2);

        assert_eq!(c_standard.shape(), c_blocked.shape());

        // Check all elements match
        for i in 0..c_standard.shape()[0] {
            for j in 0..c_standard.shape()[1] {
                assert_eq!(c_standard[[i, j]], c_blocked[[i, j]]);
            }
        }
    }

    #[test]
    fn test_khatri_rao_blocked_various_block_sizes() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let b = array![[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]];

        let c_standard = khatri_rao(&a.view(), &b.view());

        // Test different block sizes
        for block_size in [1, 2, 3, 5, 10] {
            let c_blocked = khatri_rao_blocked(&a.view(), &b.view(), block_size);

            assert_eq!(c_standard.shape(), c_blocked.shape());

            for i in 0..c_standard.shape()[0] {
                for j in 0..c_standard.shape()[1] {
                    assert_eq!(c_standard[[i, j]], c_blocked[[i, j]]);
                }
            }
        }
    }

    #[test]
    fn test_khatri_rao_blocked_small_matrix() {
        // Small matrix should fall back to standard implementation
        let a = array![[1.0, 2.0]];
        let b = array![[3.0, 4.0]];

        let c_standard = khatri_rao(&a.view(), &b.view());
        let c_blocked = khatri_rao_blocked(&a.view(), &b.view(), 2);

        assert_eq!(c_standard.shape(), c_blocked.shape());

        for i in 0..c_standard.shape()[0] {
            for j in 0..c_standard.shape()[1] {
                assert_eq!(c_standard[[i, j]], c_blocked[[i, j]]);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_khatri_rao_blocked_parallel() {
        let a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0]
        ];
        let b = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let c_standard = khatri_rao(&a.view(), &b.view());
        let c_blocked_parallel = khatri_rao_blocked_parallel(&a.view(), &b.view(), 3);

        assert_eq!(c_standard.shape(), c_blocked_parallel.shape());

        for i in 0..c_standard.shape()[0] {
            for j in 0..c_standard.shape()[1] {
                assert_eq!(c_standard[[i, j]], c_blocked_parallel[[i, j]]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Block size must be greater than 0")]
    fn test_khatri_rao_blocked_zero_block_size() {
        let a = array![[1.0, 2.0]];
        let b = array![[3.0, 4.0]];
        khatri_rao_blocked(&a.view(), &b.view(), 0);
    }

    #[test]
    #[should_panic(expected = "Number of columns must match")]
    fn test_khatri_rao_blocked_mismatched_columns() {
        let a = array![[1.0, 2.0, 3.0]];
        let b = array![[4.0, 5.0]];
        khatri_rao_blocked(&a.view(), &b.view(), 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_khatri_rao_basic() {
        // Simple 2×2 example
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = khatri_rao(&a.view(), &b.view());

        assert_eq!(c.shape(), &[4, 2]);

        // First column: [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
        assert_eq!(c[[0, 0]], 5.0);
        assert_eq!(c[[1, 0]], 7.0);
        assert_eq!(c[[2, 0]], 15.0);
        assert_eq!(c[[3, 0]], 21.0);

        // Second column: [2*6, 2*8, 4*6, 4*8] = [12, 16, 24, 32]
        assert_eq!(c[[0, 1]], 12.0);
        assert_eq!(c[[1, 1]], 16.0);
        assert_eq!(c[[2, 1]], 24.0);
        assert_eq!(c[[3, 1]], 32.0);
    }

    #[test]
    fn test_khatri_rao_different_row_sizes() {
        // A: 3×2, B: 2×2 => Result: 6×2
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0]];
        let c = khatri_rao(&a.view(), &b.view());

        assert_eq!(c.shape(), &[6, 2]);

        // First column: [1*7, 1*9, 3*7, 3*9, 5*7, 5*9]
        assert_eq!(c[[0, 0]], 7.0);
        assert_eq!(c[[1, 0]], 9.0);
        assert_eq!(c[[2, 0]], 21.0);
        assert_eq!(c[[3, 0]], 27.0);
        assert_eq!(c[[4, 0]], 35.0);
        assert_eq!(c[[5, 0]], 45.0);
    }

    #[test]
    fn test_khatri_rao_single_column() {
        // Test with single column matrices
        let a = array![[1.0], [2.0], [3.0]];
        let b = array![[4.0], [5.0]];
        let c = khatri_rao(&a.view(), &b.view());

        assert_eq!(c.shape(), &[6, 1]);
        assert_eq!(c[[0, 0]], 4.0); // 1*4
        assert_eq!(c[[1, 0]], 5.0); // 1*5
        assert_eq!(c[[2, 0]], 8.0); // 2*4
        assert_eq!(c[[3, 0]], 10.0); // 2*5
        assert_eq!(c[[4, 0]], 12.0); // 3*4
        assert_eq!(c[[5, 0]], 15.0); // 3*5
    }

    #[test]
    fn test_khatri_rao_identity_pattern() {
        // Test with identity-like matrices
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[2.0, 0.0], [0.0, 3.0]];
        let c = khatri_rao(&a.view(), &b.view());

        assert_eq!(c.shape(), &[4, 2]);

        // First column: [1*2, 1*0, 0*2, 0*0]
        assert_eq!(c[[0, 0]], 2.0);
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[2, 0]], 0.0);
        assert_eq!(c[[3, 0]], 0.0);

        // Second column: [0*0, 0*3, 1*0, 1*3]
        assert_eq!(c[[0, 1]], 0.0);
        assert_eq!(c[[1, 1]], 0.0);
        assert_eq!(c[[2, 1]], 0.0);
        assert_eq!(c[[3, 1]], 3.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_khatri_rao_parallel() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let c_serial = khatri_rao(&a.view(), &b.view());
        let c_parallel = khatri_rao_parallel(&a.view(), &b.view());

        assert_eq!(c_serial.shape(), c_parallel.shape());

        // Check all elements match
        for i in 0..c_serial.shape()[0] {
            for j in 0..c_serial.shape()[1] {
                assert_eq!(c_serial[[i, j]], c_parallel[[i, j]]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Number of columns must match")]
    fn test_khatri_rao_mismatched_columns() {
        let a = array![[1.0, 2.0, 3.0]]; // 1×3
        let b = array![[4.0, 5.0]]; // 1×2
        khatri_rao(&a.view(), &b.view()); // Should panic
    }
}
