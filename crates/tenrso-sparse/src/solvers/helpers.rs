//! Shared numerical helpers for iterative solvers
//!
//! Private to the `solvers` module. These small routines (dot, norm, axpy,
//! sparse matvec wrappers) are used by every solver implementation and are
//! kept here to avoid duplication.

use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::ndarray_ext::ArrayView1;
use scirs2_core::numeric::Float;

/// Sparse matrix-transpose-vector product: y = A^T * x
#[inline]
pub(super) fn spmv_transpose<T: Float>(matrix: &CsrMatrix<T>, x: &[T]) -> SparseResult<Vec<T>> {
    let (m, n) = matrix.shape();
    if x.len() != m {
        return Err(SparseError::validation("Vector size mismatch for A^T*x"));
    }

    let mut y = vec![T::zero(); n];

    for (i, &xi) in x.iter().enumerate() {
        let row_start = matrix.row_ptr()[i];
        let row_end = matrix.row_ptr()[i + 1];

        for idx in row_start..row_end {
            let j = matrix.col_indices()[idx];
            let val = matrix.values()[idx];
            y[j] = y[j] + val * xi;
        }
    }

    Ok(y)
}

/// Sparse matrix-vector product with Vec interface
#[inline]
pub(super) fn spmv_vec<T: Float>(matrix: &CsrMatrix<T>, x: &[T]) -> SparseResult<Vec<T>> {
    let x_array = ArrayView1::from(x);
    let result = matrix.spmv(&x_array)?;
    Ok(result.to_vec())
}

/// Dot product of two vectors
#[inline]
pub(super) fn dot<T: Float>(x: &[T], y: &[T]) -> T {
    x.iter()
        .zip(y.iter())
        .fold(T::zero(), |acc, (&xi, &yi)| acc + xi * yi)
}

/// Euclidean norm of a vector
#[inline]
pub(super) fn norm<T: Float>(x: &[T]) -> T {
    norm_squared(x).sqrt()
}

/// Squared Euclidean norm
#[inline]
pub(super) fn norm_squared<T: Float>(x: &[T]) -> T {
    x.iter().fold(T::zero(), |acc, &xi| acc + xi * xi)
}

/// AXPY: y = alpha * x + y
#[inline]
pub(super) fn axpy<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = *yi + alpha * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helper_functions() {
        let x = vec![3.0, 4.0];
        assert_eq!(norm_squared(&x), 25.0);
        assert_eq!(norm(&x), 5.0);
        assert_eq!(dot(&x, &x), 25.0);

        let mut y = vec![1.0, 2.0];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![7.0, 10.0]);
    }
}
