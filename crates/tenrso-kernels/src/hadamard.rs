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
}
