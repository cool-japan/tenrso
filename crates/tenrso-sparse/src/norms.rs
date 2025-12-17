//! # Sparse Tensor Norms
//!
//! This module provides various norm computations for sparse tensors and matrices.
//!
//! ## Available Norms
//!
//! - **Frobenius norm**: L2 norm of all elements (√Σ|aᵢⱼ|²)
//! - **L1 norm**: Sum of absolute values (Σ|aᵢⱼ|)
//! - **L2 norm**: Euclidean norm (√Σ|aᵢⱼ|²) - same as Frobenius for matrices
//! - **Infinity norm**: Maximum absolute value (max|aᵢⱼ|)
//! - **Axis-wise norms**: L1, L2, infinity norms along specified axes
//!
//! ## Complexity
//!
//! All norm operations are O(nnz) where nnz is the number of non-zero elements,
//! as they only need to iterate over stored values.
//!
//! ## Examples
//!
//! ```
//! use tenrso_sparse::{CsrMatrix, norms};
//! use tenrso_core::DenseND;
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create a simple sparse matrix
//! let data = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
//! let dense = DenseND::from_array(data.into_dyn());
//! let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
//!
//! // Compute various norms
//! let frobenius = norms::frobenius_norm_csr(&csr);
//! let l1 = norms::l1_norm_csr(&csr);
//! let linf = norms::infinity_norm_csr(&csr);
//!
//! println!("Frobenius norm: {}", frobenius);
//! println!("L1 norm: {}", l1);
//! println!("L∞ norm: {}", linf);
//! ```

use crate::{CooTensor, CscMatrix, CsrMatrix};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::numeric::Float;

/// Computes the Frobenius norm of a CSR matrix.
///
/// The Frobenius norm is defined as: ‖A‖_F = √(Σᵢⱼ |aᵢⱼ|²)
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, norms};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[3.0, 0.0], [0.0, 4.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
/// let norm = norms::frobenius_norm_csr(&csr);
/// assert!((norm - 5.0_f64).abs() < 1e-10);
/// ```
pub fn frobenius_norm_csr<T: Float>(matrix: &CsrMatrix<T>) -> T {
    matrix
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc + val * val)
        .sqrt()
}

/// Computes the Frobenius norm of a CSC matrix.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn frobenius_norm_csc<T: Float>(matrix: &CscMatrix<T>) -> T {
    matrix
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc + val * val)
        .sqrt()
}

/// Computes the Frobenius norm of a COO tensor.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn frobenius_norm_coo<T: Float>(tensor: &CooTensor<T>) -> T {
    tensor
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc + val * val)
        .sqrt()
}

/// Computes the L1 norm (sum of absolute values) of a CSR matrix.
///
/// The L1 norm is defined as: ‖A‖_1 = Σᵢⱼ |aᵢⱼ|
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, norms};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, -2.0], [3.0, -4.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
/// let norm = norms::l1_norm_csr(&csr);
/// assert!((norm - 10.0_f64).abs() < 1e-10);
/// ```
pub fn l1_norm_csr<T: Float>(matrix: &CsrMatrix<T>) -> T {
    matrix
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc + val.abs())
}

/// Computes the L1 norm of a CSC matrix.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn l1_norm_csc<T: Float>(matrix: &CscMatrix<T>) -> T {
    matrix
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc + val.abs())
}

/// Computes the L1 norm of a COO tensor.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn l1_norm_coo<T: Float>(tensor: &CooTensor<T>) -> T {
    tensor
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc + val.abs())
}

/// Computes the L2 norm (Euclidean norm) of a CSR matrix.
///
/// For matrices, this is equivalent to the Frobenius norm.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn l2_norm_csr<T: Float>(matrix: &CsrMatrix<T>) -> T {
    frobenius_norm_csr(matrix)
}

/// Computes the L2 norm of a CSC matrix.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn l2_norm_csc<T: Float>(matrix: &CscMatrix<T>) -> T {
    frobenius_norm_csc(matrix)
}

/// Computes the L2 norm of a COO tensor.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn l2_norm_coo<T: Float>(tensor: &CooTensor<T>) -> T {
    frobenius_norm_coo(tensor)
}

/// Computes the infinity norm (maximum absolute value) of a CSR matrix.
///
/// The infinity norm is defined as: ‖A‖_∞ = maxᵢⱼ |aᵢⱼ|
///
/// Note: For sparse matrices with implicit zeros, this may return zero if all
/// stored values are zero.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, norms};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, -5.0], [3.0, 2.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
/// let norm = norms::infinity_norm_csr(&csr);
/// assert!((norm - 5.0_f64).abs() < 1e-10);
/// ```
pub fn infinity_norm_csr<T: Float>(matrix: &CsrMatrix<T>) -> T {
    matrix
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc.max(val.abs()))
}

/// Computes the infinity norm of a CSC matrix.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn infinity_norm_csc<T: Float>(matrix: &CscMatrix<T>) -> T {
    matrix
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc.max(val.abs()))
}

/// Computes the infinity norm of a COO tensor.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn infinity_norm_coo<T: Float>(tensor: &CooTensor<T>) -> T {
    tensor
        .values()
        .iter()
        .fold(T::zero(), |acc, &val| acc.max(val.abs()))
}

/// Computes the matrix 1-norm (maximum absolute column sum) of a CSR matrix.
///
/// The matrix 1-norm is defined as: ‖A‖₁ = maxⱼ Σᵢ |aᵢⱼ|
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, norms};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
/// let norm = norms::matrix_1_norm_csr(&csr);
/// assert!((norm - 6.0_f64).abs() < 1e-10); // max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
/// ```
pub fn matrix_1_norm_csr<T: Float>(matrix: &CsrMatrix<T>) -> T {
    let ncols = matrix.ncols();
    let mut col_sums = vec![T::zero(); ncols];

    for (&col_idx, &val) in matrix.col_indices().iter().zip(matrix.values().iter()) {
        col_sums[col_idx] = col_sums[col_idx] + val.abs();
    }

    col_sums.iter().fold(T::zero(), |acc, &sum| acc.max(sum))
}

/// Computes the matrix 1-norm of a CSC matrix (more efficient than CSR).
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn matrix_1_norm_csc<T: Float>(matrix: &CscMatrix<T>) -> T {
    let col_ptr = matrix.col_ptr();
    let values = matrix.values();

    (0..matrix.ncols())
        .map(|j| {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            values[start..end]
                .iter()
                .fold(T::zero(), |acc, &val| acc + val.abs())
        })
        .fold(T::zero(), |acc, sum| acc.max(sum))
}

/// Computes the matrix infinity-norm (maximum absolute row sum) of a CSR matrix.
///
/// The matrix infinity-norm is defined as: ‖A‖_∞ = maxᵢ Σⱼ |aᵢⱼ|
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, norms};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
/// let norm = norms::matrix_infinity_norm_csr(&csr);
/// assert!((norm - 7.0_f64).abs() < 1e-10); // max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
/// ```
pub fn matrix_infinity_norm_csr<T: Float>(matrix: &CsrMatrix<T>) -> T {
    let row_ptr = matrix.row_ptr();
    let values = matrix.values();

    (0..matrix.nrows())
        .map(|i| {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            values[start..end]
                .iter()
                .fold(T::zero(), |acc, &val| acc + val.abs())
        })
        .fold(T::zero(), |acc, sum| acc.max(sum))
}

/// Computes the matrix infinity-norm of a CSC matrix.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements.
pub fn matrix_infinity_norm_csc<T: Float>(matrix: &CscMatrix<T>) -> T {
    let nrows = matrix.nrows();
    let mut row_sums = vec![T::zero(); nrows];

    for (&row_idx, &val) in matrix.row_indices().iter().zip(matrix.values().iter()) {
        row_sums[row_idx] = row_sums[row_idx] + val.abs();
    }

    row_sums.iter().fold(T::zero(), |acc, &sum| acc.max(sum))
}

/// Computes L1 norm along a specified axis for CSR matrix.
///
/// - `axis = 0`: Column-wise L1 norms (sum of absolute values per column)
/// - `axis = 1`: Row-wise L1 norms (sum of absolute values per row)
///
/// # Complexity
///
/// O(nnz) for axis=1 (row-wise), O(nnz + ncols) for axis=0 (column-wise)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, norms};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let row_norms = norms::l1_norm_axis_csr(&csr, 1);
/// assert!((row_norms[0] - 3.0_f64).abs() < 1e-10); // |1| + |2| = 3
/// assert!((row_norms[1] - 7.0_f64).abs() < 1e-10); // |3| + |4| = 7
/// ```
pub fn l1_norm_axis_csr<T: Float>(matrix: &CsrMatrix<T>, axis: usize) -> Array1<T> {
    match axis {
        0 => {
            // Column-wise norms
            let mut col_norms = Array1::zeros(matrix.ncols());
            for (&col_idx, &val) in matrix.col_indices().iter().zip(matrix.values().iter()) {
                col_norms[col_idx] = col_norms[col_idx] + val.abs();
            }
            col_norms
        }
        1 => {
            // Row-wise norms
            let row_ptr = matrix.row_ptr();
            let values = matrix.values();
            let mut row_norms = Array1::zeros(matrix.nrows());
            for i in 0..matrix.nrows() {
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                row_norms[i] = values[start..end]
                    .iter()
                    .fold(T::zero(), |acc, &val| acc + val.abs());
            }
            row_norms
        }
        _ => panic!("Invalid axis {} for 2D matrix, must be 0 or 1", axis),
    }
}

/// Computes L2 norm along a specified axis for CSR matrix.
///
/// - `axis = 0`: Column-wise L2 norms
/// - `axis = 1`: Row-wise L2 norms
///
/// # Complexity
///
/// O(nnz) for axis=1 (row-wise), O(nnz + ncols) for axis=0 (column-wise)
pub fn l2_norm_axis_csr<T: Float>(matrix: &CsrMatrix<T>, axis: usize) -> Array1<T> {
    match axis {
        0 => {
            // Column-wise norms
            let mut col_norms = Array1::<T>::zeros(matrix.ncols());
            for (&col_idx, &val) in matrix.col_indices().iter().zip(matrix.values().iter()) {
                col_norms[col_idx] = col_norms[col_idx] + val * val;
            }
            for x in col_norms.iter_mut() {
                *x = x.sqrt();
            }
            col_norms
        }
        1 => {
            // Row-wise norms
            let row_ptr = matrix.row_ptr();
            let values = matrix.values();
            let mut row_norms = Array1::zeros(matrix.nrows());
            for i in 0..matrix.nrows() {
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                row_norms[i] = values[start..end]
                    .iter()
                    .fold(T::zero(), |acc, &val| acc + val * val)
                    .sqrt();
            }
            row_norms
        }
        _ => panic!("Invalid axis {} for 2D matrix, must be 0 or 1", axis),
    }
}

/// Computes infinity norm along a specified axis for CSR matrix.
///
/// - `axis = 0`: Column-wise infinity norms (max absolute value per column)
/// - `axis = 1`: Row-wise infinity norms (max absolute value per row)
///
/// # Complexity
///
/// O(nnz) for axis=1 (row-wise), O(nnz + ncols) for axis=0 (column-wise)
pub fn infinity_norm_axis_csr<T: Float>(matrix: &CsrMatrix<T>, axis: usize) -> Array1<T> {
    match axis {
        0 => {
            // Column-wise norms
            let mut col_norms = Array1::<T>::zeros(matrix.ncols());
            for (&col_idx, &val) in matrix.col_indices().iter().zip(matrix.values().iter()) {
                col_norms[col_idx] = col_norms[col_idx].max(val.abs());
            }
            col_norms
        }
        1 => {
            // Row-wise norms
            let row_ptr = matrix.row_ptr();
            let values = matrix.values();
            let mut row_norms = Array1::zeros(matrix.nrows());
            for i in 0..matrix.nrows() {
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                row_norms[i] = values[start..end]
                    .iter()
                    .fold(T::zero(), |acc, &val| acc.max(val.abs()));
            }
            row_norms
        }
        _ => panic!("Invalid axis {} for 2D matrix, must be 0 or 1", axis),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_frobenius_norm_csr() {
        use tenrso_core::DenseND;
        let data = array![[3.0, 0.0], [0.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let norm = frobenius_norm_csr(&csr);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm_csr() {
        use tenrso_core::DenseND;
        let data = array![[1.0, -2.0], [3.0, -4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let norm = l1_norm_csr(&csr);
        assert!((norm - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_infinity_norm_csr() {
        use tenrso_core::DenseND;
        let data = array![[1.0, -5.0], [3.0, 2.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let norm = infinity_norm_csr(&csr);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_1_norm_csr() {
        use tenrso_core::DenseND;
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let norm = matrix_1_norm_csr(&csr);
        assert!((norm - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_infinity_norm_csr() {
        use tenrso_core::DenseND;
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let norm = matrix_infinity_norm_csr(&csr);
        assert!((norm - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm_axis_csr() {
        use tenrso_core::DenseND;
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let row_norms = l1_norm_axis_csr(&csr, 1);
        assert!((row_norms[0] - 3.0).abs() < 1e-10);
        assert!((row_norms[1] - 7.0).abs() < 1e-10);

        let col_norms = l1_norm_axis_csr(&csr, 0);
        assert!((col_norms[0] - 4.0).abs() < 1e-10);
        assert!((col_norms[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm_axis_csr() {
        use tenrso_core::DenseND;
        let data = array![[3.0, 0.0], [0.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let row_norms = l2_norm_axis_csr(&csr, 1);
        assert!((row_norms[0] - 3.0).abs() < 1e-10);
        assert!((row_norms[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_infinity_norm_axis_csr() {
        use tenrso_core::DenseND;
        let data = array![[1.0, 5.0], [3.0, 2.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let row_norms = infinity_norm_axis_csr(&csr, 1);
        assert!((row_norms[0] - 5.0).abs() < 1e-10);
        assert!((row_norms[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_norms_csc() {
        use tenrso_core::DenseND;
        let data = array![[3.0, 0.0], [0.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 0.0).unwrap();

        let frobenius = frobenius_norm_csc(&csc);
        assert!((frobenius - 5.0).abs() < 1e-10);

        let l1 = l1_norm_csc(&csc);
        assert!((l1 - 7.0).abs() < 1e-10);

        let linf = infinity_norm_csc(&csc);
        assert!((linf - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_norms_csc() {
        use tenrso_core::DenseND;
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 0.0).unwrap();

        let norm1 = matrix_1_norm_csc(&csc);
        assert!((norm1 - 6.0).abs() < 1e-10);

        let norminf = matrix_infinity_norm_csc(&csc);
        assert!((norminf - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_norms_coo() {
        use tenrso_core::DenseND;
        let data = array![[3.0, 0.0], [0.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let coo = csr.to_coo();

        let frobenius = frobenius_norm_coo(&coo);
        assert!((frobenius - 5.0).abs() < 1e-10);

        let l1 = l1_norm_coo(&coo);
        assert!((l1 - 7.0).abs() < 1e-10);

        let linf = infinity_norm_coo(&coo);
        assert!((linf - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_matrix_norms() {
        let empty = CsrMatrix::<f64>::new(vec![0, 0, 0, 0], vec![], vec![], (3, 3)).unwrap();

        assert_eq!(frobenius_norm_csr(&empty), 0.0);
        assert_eq!(l1_norm_csr(&empty), 0.0);
        assert_eq!(infinity_norm_csr(&empty), 0.0);
        assert_eq!(matrix_1_norm_csr(&empty), 0.0);
        assert_eq!(matrix_infinity_norm_csr(&empty), 0.0);
    }

    #[test]
    fn test_sparse_pattern_norms() {
        use tenrso_core::DenseND;
        // Matrix with specific sparse pattern
        let data = array![
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [5.0, 0.0, 0.0, 6.0]
        ];
        let dense = DenseND::from_array(data.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        // Frobenius: sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91)
        let frobenius = frobenius_norm_csr(&csr);
        assert!((frobenius - 91.0_f64.sqrt()).abs() < 1e-10);

        // L1: 1 + 2 + 3 + 4 + 5 + 6 = 21
        let l1 = l1_norm_csr(&csr);
        assert!((l1 - 21.0).abs() < 1e-10);

        // L∞: max(1, 2, 3, 4, 5, 6) = 6
        let linf = infinity_norm_csr(&csr);
        assert!((linf - 6.0).abs() < 1e-10);

        // Matrix 1-norm: max column sum = max(6, 3, 4, 8) = 8
        let norm1 = matrix_1_norm_csr(&csr);
        assert!((norm1 - 8.0).abs() < 1e-10);

        // Matrix ∞-norm: max row sum = max(3, 3, 4, 11) = 11
        let norminf = matrix_infinity_norm_csr(&csr);
        assert!((norminf - 11.0).abs() < 1e-10);
    }
}
