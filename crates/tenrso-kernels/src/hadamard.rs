//! Hadamard (element-wise) product implementation
//!
//! The Hadamard product is element-wise multiplication of tensors.
//! For tensors A and B of the same shape, C = A ⊙ B where c_ijk... = a_ijk... * b_ijk...
//!
//! This is a fundamental operation in neural networks and tensor computations.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use scirs2_core::ndarray_ext::{Array, Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::Num;

/// Compute the Hadamard (element-wise) product of two matrices
///
/// For matrices A and B of the same shape, C = A ⊙ B where c_ij = a_ij * b_ij.
///
/// # Arguments
///
/// * `a` - First matrix with shape (m, n)
/// * `b` - Second matrix with shape (m, n)
///
/// # Returns
///
/// A matrix with shape (m, n) containing the element-wise product
///
/// # Panics
///
/// Panics if the shapes of A and B don't match
///
/// # Complexity
///
/// Time: O(m * n)
/// Space: O(m * n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::hadamard;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let c = hadamard(&a.view(), &b.view());
/// assert_eq!(c.shape(), &[2, 2]);
/// assert_eq!(c[[0, 0]], 5.0);   // 1*5
/// assert_eq!(c[[0, 1]], 12.0);  // 2*6
/// assert_eq!(c[[1, 0]], 21.0);  // 3*7
/// assert_eq!(c[[1, 1]], 32.0);  // 4*8
/// ```
pub fn hadamard<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Num,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Shapes must match for Hadamard product: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );

    // Element-wise multiplication
    a * b
}

/// Compute the Hadamard (element-wise) product of two N-dimensional tensors
///
/// For tensors A and B of the same shape, C = A ⊙ B where c_ijk... = a_ijk... * b_ijk...
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor with the same shape as A
///
/// # Returns
///
/// A tensor with the same shape containing the element-wise product
///
/// # Panics
///
/// Panics if the shapes of A and B don't match
///
/// # Complexity
///
/// Time: O(total elements)
/// Space: O(total elements)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::hadamard_nd;
///
/// let a = Array::from_shape_vec(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
/// let b = Array::from_shape_vec(vec![2, 2, 2], vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).unwrap();
/// let c = hadamard_nd(&a.view(), &b.view());
/// assert_eq!(c.shape(), &[2, 2, 2]);
/// ```
pub fn hadamard_nd<T>(a: &ArrayView<T, IxDyn>, b: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Num,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Shapes must match for Hadamard product: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );

    // Element-wise multiplication
    a * b
}

/// In-place Hadamard (element-wise) product for matrices
///
/// Computes `a = a ⊙ b` in-place, modifying matrix `a`.
///
/// # Arguments
///
/// * `a` - Mutable matrix that will be modified in-place
/// * `b` - Second matrix with the same shape as A
///
/// # Panics
///
/// Panics if the shapes of A and B don't match
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::array;
/// use tenrso_kernels::hadamard_inplace;
///
/// let mut a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// hadamard_inplace(&mut a.view_mut(), &b.view());
/// assert_eq!(a[[0, 0]], 5.0);
/// assert_eq!(a[[0, 1]], 12.0);
/// ```
pub fn hadamard_inplace<T>(a: &mut scirs2_core::ndarray_ext::ArrayViewMut2<T>, b: &ArrayView2<T>)
where
    T: Clone + Num,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Shapes must match for Hadamard product: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );

    // In-place element-wise multiplication using Zip
    use scirs2_core::ndarray_ext::Zip;
    Zip::from(&mut *a).and(b).for_each(|a_elem, b_elem| {
        *a_elem = a_elem.clone() * b_elem.clone();
    });
}

/// Row-parallel Hadamard (element-wise) product for matrices.
///
/// Partitions the output by rows across Rayon threads; each thread computes
/// its row(s) with a plain scalar loop (no cross-thread synchronization in
/// the hot loop).
///
/// # Performance note
///
/// Hadamard is a single multiply-and-store per element: one FMA-free op for
/// two loads and one store. On most hardware this is **memory-bandwidth
/// bound**, not compute bound, which is exactly why the original TODO for
/// this function flagged it as only a "small benefit" candidate — see the
/// benchmark results in `benches/kernel_benchmarks.rs` (`hadamard` group,
/// `parallel` entries) for the measured numbers on this machine: a genuine
/// ~2.4x speedup at 2000x2000 (8-core Xeon), but ~30x *slower* than serial
/// at 100x100 (Rayon fork/join overhead dominates below roughly 500x500).
/// Prefer [`hadamard`] for small matrices; switch to this function only
/// once you've benchmarked a win for your matrix sizes and thread count.
///
/// # Panics
///
/// Panics if the shapes of A and B don't match.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_kernels::hadamard_parallel;
///
/// let a = Array2::<f64>::ones((64, 64));
/// let b = Array2::<f64>::ones((64, 64)) * 2.0;
/// let c = hadamard_parallel(&a.view(), &b.view());
/// assert_eq!(c[[0, 0]], 2.0);
/// ```
#[cfg(feature = "parallel")]
pub fn hadamard_parallel<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Num + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    assert_eq!(
        a.shape(),
        b.shape(),
        "Shapes must match for Hadamard product: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );

    let (rows, cols) = a.dim();

    // Each thread builds its row with `.collect()` (single pass, no
    // pre-zeroing); rows are flattened into the final buffer afterwards.
    // Deliberately *not* `Array2::zeros` + `axis_iter_mut` — that pattern
    // touches every output element twice (a zero-fill pass, then the real
    // write), doubling memory traffic for what is an inherently
    // bandwidth-bound op.
    let rows_data: Vec<Vec<T>> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let a_row = a.row(i);
            let b_row = b.row(i);
            (0..cols)
                .map(|j| a_row[j].clone() * b_row[j].clone())
                .collect()
        })
        .collect();

    let flat: Vec<T> = rows_data.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, cols), flat).expect("flat.len() == rows * cols by construction")
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_hadamard_basic() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = hadamard(&a.view(), &b.view());

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[[0, 0]], 5.0); // 1*5
        assert_eq!(c[[0, 1]], 12.0); // 2*6
        assert_eq!(c[[1, 0]], 21.0); // 3*7
        assert_eq!(c[[1, 1]], 32.0); // 4*8
    }

    #[test]
    fn test_hadamard_zeros() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.0, 0.0], [0.0, 0.0]];
        let c = hadamard(&a.view(), &b.view());

        assert_eq!(c[[0, 0]], 0.0);
        assert_eq!(c[[0, 1]], 0.0);
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[1, 1]], 0.0);
    }

    #[test]
    fn test_hadamard_identity() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let ones = array![[1.0, 1.0], [1.0, 1.0]];
        let c = hadamard(&a.view(), &ones.view());

        // Should be unchanged
        assert_eq!(c[[0, 0]], 1.0);
        assert_eq!(c[[0, 1]], 2.0);
        assert_eq!(c[[1, 0]], 3.0);
        assert_eq!(c[[1, 1]], 4.0);
    }

    #[test]
    fn test_hadamard_nd() {
        // 3D tensor example
        let a = Array::from_shape_vec(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let b = Array::from_shape_vec(vec![2, 2, 2], vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            .unwrap();

        let c = hadamard_nd(&a.view(), &b.view());

        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(c[[0, 0, 0]], 2.0); // 1*2
        assert_eq!(c[[0, 0, 1]], 4.0); // 2*2
        assert_eq!(c[[1, 1, 1]], 16.0); // 8*2
    }

    #[test]
    fn test_hadamard_inplace() {
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        hadamard_inplace(&mut a.view_mut(), &b.view());

        assert_eq!(a[[0, 0]], 5.0);
        assert_eq!(a[[0, 1]], 12.0);
        assert_eq!(a[[1, 0]], 21.0);
        assert_eq!(a[[1, 1]], 32.0);
    }

    #[test]
    #[should_panic(expected = "Shapes must match")]
    fn test_hadamard_mismatched_shapes() {
        let a = array![[1.0, 2.0]]; // 1×2
        let b = array![[3.0], [4.0]]; // 2×1
        hadamard(&a.view(), &b.view()); // Should panic
    }

    #[test]
    fn test_hadamard_single_element() {
        let a = array![[5.0]];
        let b = array![[3.0]];
        let c = hadamard(&a.view(), &b.view());

        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c[[0, 0]], 15.0);
    }

    #[test]
    fn test_hadamard_large() {
        // Test with larger matrices
        let a = Array2::<f64>::ones((100, 100)) * 2.0;
        let b = Array2::<f64>::ones((100, 100)) * 3.0;
        let c = hadamard(&a.view(), &b.view());

        assert_eq!(c.shape(), &[100, 100]);
        assert_eq!(c[[0, 0]], 6.0);
        assert_eq!(c[[99, 99]], 6.0);
    }

    // Sizes below straddle small/edge cases (0, 1, non-power-of-two, etc.) to
    // exercise `hadamard_parallel`'s row-partitioning at boundary conditions.
    const SIMD_TEST_SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 32, 33, 65, 129];

    #[cfg(feature = "parallel")]
    #[test]
    fn test_hadamard_parallel_matches_scalar_all_sizes() {
        for &n in SIMD_TEST_SIZES {
            let a = Array2::<f64>::from_shape_fn((5, n), |(i, j)| (i * 3 + j + 1) as f64);
            let b = Array2::<f64>::from_shape_fn((5, n), |(i, j)| (i + j * 2 + 1) as f64 * 0.3);

            let scalar = hadamard(&a.view(), &b.view());
            let parallel = hadamard_parallel(&a.view(), &b.view());

            assert_eq!(
                scalar.shape(),
                parallel.shape(),
                "shape mismatch at n={}",
                n
            );
            for (x, y) in scalar.iter().zip(parallel.iter()) {
                assert!((x - y).abs() < 1e-12, "mismatch at n={}: {} vs {}", n, x, y);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[should_panic(expected = "Shapes must match")]
    fn test_hadamard_parallel_mismatched_shapes() {
        let a = array![[1.0, 2.0]];
        let b = array![[3.0], [4.0]];
        hadamard_parallel(&a.view(), &b.view());
    }
}
