//! # Sparse Tensor Structural Operations
//!
//! This module provides structural operations for sparse matrices and tensors,
//! including:
//!
//! - **Stacking**: Vertical (vstack) and horizontal (hstack) concatenation
//! - **Diagonal operations**: Extract diagonal, construct diagonal matrix
//! - **Triangular operations**: Extract upper/lower triangular portions
//! - **Block operations**: Block diagonal construction
//!
//! ## Examples
//!
//! ```
//! use tenrso_sparse::{CsrMatrix, structural};
//! use tenrso_core::DenseND;
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create two sparse matrices
//! let data1 = array![[1.0, 2.0], [3.0, 4.0]];
//! let dense1 = DenseND::from_array(data1.into_dyn());
//! let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();
//!
//! let data2 = array![[5.0, 6.0], [7.0, 8.0]];
//! let dense2 = DenseND::from_array(data2.into_dyn());
//! let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();
//!
//! // Vertical stack
//! let vstacked = structural::vstack_csr(&[&mat1, &mat2]).unwrap();
//! assert_eq!(vstacked.nrows(), 4);
//! assert_eq!(vstacked.ncols(), 2);
//! ```

use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::numeric::Float;

/// Vertically stacks (concatenates) CSR matrices.
///
/// Combines multiple CSR matrices by stacking them vertically (row-wise).
/// All matrices must have the same number of columns.
///
/// # Complexity
///
/// O(Σ nnz_i) where nnz_i is the number of non-zeros in matrix i.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, structural};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let dense1 = DenseND::from_array(data1.into_dyn());
/// let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();
///
/// let data2 = array![[5.0, 6.0]];
/// let dense2 = DenseND::from_array(data2.into_dyn());
/// let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();
///
/// let result = structural::vstack_csr(&[&mat1, &mat2]).unwrap();
/// assert_eq!(result.nrows(), 3);
/// assert_eq!(result.ncols(), 2);
/// ```
pub fn vstack_csr<T: Float>(matrices: &[&CsrMatrix<T>]) -> SparseResult<CsrMatrix<T>> {
    if matrices.is_empty() {
        return Err(SparseError::validation(
            "Cannot vstack empty list of matrices",
        ));
    }

    let ncols = matrices[0].ncols();

    // Check all matrices have same number of columns
    for (i, mat) in matrices.iter().enumerate().skip(1) {
        if mat.ncols() != ncols {
            return Err(SparseError::validation(&format!(
                "Matrix {} has {} columns, expected {}",
                i,
                mat.ncols(),
                ncols
            )));
        }
    }

    let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
    let total_nnz: usize = matrices.iter().map(|m| m.nnz()).sum();

    let mut row_ptr = Vec::with_capacity(total_rows + 1);
    let mut col_indices = Vec::with_capacity(total_nnz);
    let mut values = Vec::with_capacity(total_nnz);

    row_ptr.push(0);

    for matrix in matrices {
        let mat_row_ptr = matrix.row_ptr();
        let mat_col_indices = matrix.col_indices();
        let mat_values = matrix.values();

        // Append rows from this matrix
        for i in 0..matrix.nrows() {
            let start = mat_row_ptr[i];
            let end = mat_row_ptr[i + 1];

            col_indices.extend_from_slice(&mat_col_indices[start..end]);
            values.extend_from_slice(&mat_values[start..end]);

            row_ptr.push(col_indices.len());
        }
    }

    CsrMatrix::new(row_ptr, col_indices, values, (total_rows, ncols))
        .map_err(|_| SparseError::operation("Failed to create vstacked CSR matrix"))
}

/// Horizontally stacks (concatenates) CSR matrices.
///
/// Combines multiple CSR matrices by stacking them horizontally (column-wise).
/// All matrices must have the same number of rows.
///
/// # Complexity
///
/// O(Σ nnz_i) where nnz_i is the number of non-zeros in matrix i.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, structural};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data1 = array![[1.0], [3.0]];
/// let dense1 = DenseND::from_array(data1.into_dyn());
/// let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();
///
/// let data2 = array![[2.0], [4.0]];
/// let dense2 = DenseND::from_array(data2.into_dyn());
/// let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();
///
/// let result = structural::hstack_csr(&[&mat1, &mat2]).unwrap();
/// assert_eq!(result.nrows(), 2);
/// assert_eq!(result.ncols(), 2);
/// ```
pub fn hstack_csr<T: Float>(matrices: &[&CsrMatrix<T>]) -> SparseResult<CsrMatrix<T>> {
    if matrices.is_empty() {
        return Err(SparseError::validation(
            "Cannot hstack empty list of matrices",
        ));
    }

    let nrows = matrices[0].nrows();

    // Check all matrices have same number of rows
    for (i, mat) in matrices.iter().enumerate().skip(1) {
        if mat.nrows() != nrows {
            return Err(SparseError::validation(&format!(
                "Matrix {} has {} rows, expected {}",
                i,
                mat.nrows(),
                nrows
            )));
        }
    }

    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
    let total_nnz: usize = matrices.iter().map(|m| m.nnz()).sum();

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::with_capacity(total_nnz);
    let mut values = Vec::with_capacity(total_nnz);

    row_ptr.push(0);

    for i in 0..nrows {
        let mut col_offset = 0;

        for matrix in matrices {
            let mat_row_ptr = matrix.row_ptr();
            let mat_col_indices = matrix.col_indices();
            let mat_values = matrix.values();

            let start = mat_row_ptr[i];
            let end = mat_row_ptr[i + 1];

            // Append elements from this row with adjusted column indices
            for j in start..end {
                col_indices.push(mat_col_indices[j] + col_offset);
                values.push(mat_values[j]);
            }

            col_offset += matrix.ncols();
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (nrows, total_cols))
        .map_err(|_| SparseError::operation("Failed to create hstacked CSR matrix"))
}

/// Extracts the diagonal of a CSR matrix as a dense array.
///
/// Returns the main diagonal (k=0) or other diagonals:
/// - k > 0: super-diagonal (above main diagonal)
/// - k < 0: sub-diagonal (below main diagonal)
///
/// # Complexity
///
/// O(nnz) in worst case, typically O(min(m,n)) for sparse matrices.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, structural};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [0.0, 8.0, 9.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let diag = structural::diagonal_csr(&mat, 0);
/// assert!((diag[0] - 1.0_f64).abs() < 1e-10);
/// assert!((diag[1] - 5.0_f64).abs() < 1e-10);
/// assert!((diag[2] - 9.0_f64).abs() < 1e-10);
/// ```
pub fn diagonal_csr<T: Float>(matrix: &CsrMatrix<T>, k: isize) -> Array1<T> {
    let nrows = matrix.nrows() as isize;
    let ncols = matrix.ncols() as isize;

    // Compute diagonal length
    let diag_len = if k >= 0 {
        std::cmp::max(0, std::cmp::min(nrows, ncols - k)) as usize
    } else {
        std::cmp::max(0, std::cmp::min(nrows + k, ncols)) as usize
    };

    let mut diagonal = Array1::<T>::zeros(diag_len);

    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();
    let values = matrix.values();

    for i in 0..diag_len {
        let row = if k >= 0 { i } else { (i as isize - k) as usize };
        let col = if k >= 0 { (i as isize + k) as usize } else { i };

        if row >= matrix.nrows() || col >= matrix.ncols() {
            break;
        }

        let start = row_ptr[row];
        let end = row_ptr[row + 1];

        // Binary search for the column index
        if let Ok(idx) = col_indices[start..end].binary_search(&col) {
            diagonal[i] = values[start + idx];
        }
        // Otherwise it remains zero (implicit zero in sparse matrix)
    }

    diagonal
}

/// Constructs a diagonal CSR matrix from a dense array.
///
/// Creates a sparse matrix with the given values on the diagonal and zeros elsewhere.
/// The optional `k` parameter specifies the diagonal offset:
/// - k = 0: main diagonal (default)
/// - k > 0: super-diagonal
/// - k < 0: sub-diagonal
///
/// # Complexity
///
/// O(n) where n is the length of the diagonal array.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::structural;
/// use scirs2_core::ndarray_ext::array;
///
/// let diag = array![1.0, 2.0, 3.0];
/// let mat = structural::diag_csr(&diag.view(), 0).unwrap();
/// assert_eq!(mat.nrows(), 3);
/// assert_eq!(mat.ncols(), 3);
/// assert_eq!(mat.nnz(), 3);
/// ```
pub fn diag_csr<T: Float>(
    diagonal: &scirs2_core::ndarray_ext::ArrayView1<T>,
    k: isize,
) -> SparseResult<CsrMatrix<T>> {
    let n = diagonal.len();
    if n == 0 {
        return Err(SparseError::validation(
            "Cannot create diagonal matrix from empty array",
        ));
    }

    let size = (n as isize + k.abs()) as usize;
    let nrows = if k >= 0 { n } else { size };
    let ncols = if k >= 0 { size } else { n };

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    row_ptr.push(0);

    for i in 0..nrows {
        let diag_idx = if k >= 0 {
            if i < n {
                Some(i)
            } else {
                None
            }
        } else {
            let diag_pos = i as isize + k;
            if diag_pos >= 0 && diag_pos < n as isize {
                Some(diag_pos as usize)
            } else {
                None
            }
        };

        if let Some(idx) = diag_idx {
            let val = diagonal[idx];
            if val != T::zero() {
                let col_isize = i as isize + k;
                if col_isize < 0 {
                    continue;
                }
                let col = col_isize as usize;
                if col < ncols {
                    col_indices.push(col);
                    values.push(val);
                }
            }
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (nrows, ncols))
        .map_err(|_| SparseError::operation("Failed to create diagonal CSR matrix"))
}

/// Extracts the upper triangular portion of a CSR matrix.
///
/// Returns a new CSR matrix containing only the upper triangular elements.
/// The optional `k` parameter controls which diagonal to include:
/// - k = 0: include main diagonal (default)
/// - k > 0: exclude k diagonals below main
/// - k < 0: include k diagonals below main
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zeros in the input matrix.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, structural};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let upper = structural::triu_csr(&mat, 0).unwrap();
/// // Should contain: 1,2,3,5,6,9
/// assert_eq!(upper.nnz(), 6);
/// ```
pub fn triu_csr<T: Float>(matrix: &CsrMatrix<T>, k: isize) -> SparseResult<CsrMatrix<T>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);

    let mat_row_ptr = matrix.row_ptr();
    let mat_col_indices = matrix.col_indices();
    let mat_values = matrix.values();

    for i in 0..nrows {
        let start = mat_row_ptr[i];
        let end = mat_row_ptr[i + 1];

        let min_col = if k >= 0 {
            (i as isize + k) as usize
        } else {
            i.saturating_sub((-k) as usize)
        };

        for j in start..end {
            let col = mat_col_indices[j];
            if col >= min_col {
                col_indices.push(col);
                values.push(mat_values[j]);
            }
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (nrows, ncols))
        .map_err(|_| SparseError::operation("Failed to create upper triangular CSR matrix"))
}

/// Extracts the lower triangular portion of a CSR matrix.
///
/// Returns a new CSR matrix containing only the lower triangular elements.
/// The optional `k` parameter controls which diagonal to include:
/// - k = 0: include main diagonal (default)
/// - k > 0: include k diagonals above main
/// - k < 0: exclude k diagonals above main
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zeros in the input matrix.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, structural};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let lower = structural::tril_csr(&mat, 0).unwrap();
/// // Should contain: 1,4,5,7,8,9
/// assert_eq!(lower.nnz(), 6);
/// ```
pub fn tril_csr<T: Float>(matrix: &CsrMatrix<T>, k: isize) -> SparseResult<CsrMatrix<T>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);

    let mat_row_ptr = matrix.row_ptr();
    let mat_col_indices = matrix.col_indices();
    let mat_values = matrix.values();

    for i in 0..nrows {
        let start = mat_row_ptr[i];
        let end = mat_row_ptr[i + 1];

        let max_col_isize = i as isize + k;

        for j in start..end {
            let col = mat_col_indices[j];
            if max_col_isize >= 0 && col <= max_col_isize as usize {
                col_indices.push(col);
                values.push(mat_values[j]);
            }
        }

        row_ptr.push(col_indices.len());
    }

    CsrMatrix::new(row_ptr, col_indices, values, (nrows, ncols))
        .map_err(|_| SparseError::operation("Failed to create lower triangular CSR matrix"))
}

/// Constructs a block diagonal CSR matrix from multiple CSR matrices.
///
/// Creates a sparse matrix with the given matrices along the diagonal and zeros elsewhere.
///
/// # Complexity
///
/// O(Σ nnz_i) where nnz_i is the number of non-zeros in matrix i.
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, structural};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let dense1 = DenseND::from_array(data1.into_dyn());
/// let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();
///
/// let data2 = array![[5.0]];
/// let dense2 = DenseND::from_array(data2.into_dyn());
/// let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();
///
/// let block_diag = structural::block_diag_csr(&[&mat1, &mat2]).unwrap();
/// assert_eq!(block_diag.nrows(), 3);
/// assert_eq!(block_diag.ncols(), 3);
/// ```
pub fn block_diag_csr<T: Float>(matrices: &[&CsrMatrix<T>]) -> SparseResult<CsrMatrix<T>> {
    if matrices.is_empty() {
        return Err(SparseError::validation(
            "Cannot create block diagonal from empty list",
        ));
    }

    let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
    let total_nnz: usize = matrices.iter().map(|m| m.nnz()).sum();

    let mut row_ptr = Vec::with_capacity(total_rows + 1);
    let mut col_indices = Vec::with_capacity(total_nnz);
    let mut values = Vec::with_capacity(total_nnz);

    row_ptr.push(0);

    let mut col_offset = 0;

    for matrix in matrices {
        let mat_row_ptr = matrix.row_ptr();
        let mat_col_indices = matrix.col_indices();
        let mat_values = matrix.values();

        for i in 0..matrix.nrows() {
            let start = mat_row_ptr[i];
            let end = mat_row_ptr[i + 1];

            for j in start..end {
                col_indices.push(mat_col_indices[j] + col_offset);
                values.push(mat_values[j]);
            }

            row_ptr.push(col_indices.len());
        }

        col_offset += matrix.ncols();
    }

    CsrMatrix::new(row_ptr, col_indices, values, (total_rows, total_cols))
        .map_err(|_| SparseError::operation("Failed to create block diagonal CSR matrix"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;
    use tenrso_core::DenseND;

    #[test]
    fn test_vstack_csr() {
        let data1 = array![[1.0, 2.0], [3.0, 4.0]];
        let dense1 = DenseND::from_array(data1.into_dyn());
        let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();

        let data2 = array![[5.0, 6.0]];
        let dense2 = DenseND::from_array(data2.into_dyn());
        let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();

        let result = vstack_csr(&[&mat1, &mat2]).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result.nnz(), 6);
    }

    #[test]
    fn test_hstack_csr() {
        let data1 = array![[1.0], [3.0]];
        let dense1 = DenseND::from_array(data1.into_dyn());
        let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();

        let data2 = array![[2.0], [4.0]];
        let dense2 = DenseND::from_array(data2.into_dyn());
        let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();

        let result = hstack_csr(&[&mat1, &mat2]).unwrap();
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result.nnz(), 4);
    }

    #[test]
    fn test_diagonal_csr() {
        let data = array![[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [0.0, 8.0, 9.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let diag = diagonal_csr(&mat, 0);
        assert_eq!(diag.len(), 3);
        assert!((diag[0] - 1.0).abs() < 1e-10);
        assert!((diag[1] - 5.0).abs() < 1e-10);
        assert!((diag[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_diag_csr() {
        let diag = array![1.0, 2.0, 3.0];
        let mat = diag_csr(&diag.view(), 0).unwrap();
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 3);
        assert_eq!(mat.nnz(), 3);
    }

    #[test]
    fn test_triu_csr() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let upper = triu_csr(&mat, 0).unwrap();
        assert_eq!(upper.nnz(), 6); // 1,2,3,5,6,9
    }

    #[test]
    fn test_tril_csr() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let lower = tril_csr(&mat, 0).unwrap();
        assert_eq!(lower.nnz(), 6); // 1,4,5,7,8,9
    }

    #[test]
    fn test_block_diag_csr() {
        let data1 = array![[1.0, 2.0], [3.0, 4.0]];
        let dense1 = DenseND::from_array(data1.into_dyn());
        let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();

        let data2 = array![[5.0]];
        let dense2 = DenseND::from_array(data2.into_dyn());
        let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();

        let block_diag = block_diag_csr(&[&mat1, &mat2]).unwrap();
        assert_eq!(block_diag.nrows(), 3);
        assert_eq!(block_diag.ncols(), 3);
        assert_eq!(block_diag.nnz(), 5);
    }

    #[test]
    fn test_vstack_shape_mismatch() {
        let data1 = array![[1.0, 2.0]];
        let dense1 = DenseND::from_array(data1.into_dyn());
        let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();

        let data2 = array![[3.0, 4.0, 5.0]];
        let dense2 = DenseND::from_array(data2.into_dyn());
        let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();

        assert!(vstack_csr(&[&mat1, &mat2]).is_err());
    }

    #[test]
    fn test_hstack_shape_mismatch() {
        let data1 = array![[1.0]];
        let dense1 = DenseND::from_array(data1.into_dyn());
        let mat1 = CsrMatrix::from_dense(&dense1, 0.0).unwrap();

        let data2 = array![[2.0], [3.0]];
        let dense2 = DenseND::from_array(data2.into_dyn());
        let mat2 = CsrMatrix::from_dense(&dense2, 0.0).unwrap();

        assert!(hstack_csr(&[&mat1, &mat2]).is_err());
    }

    #[test]
    fn test_diagonal_super_sub() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        // Super-diagonal (k=1)
        let super_diag = diagonal_csr(&mat, 1);
        assert_eq!(super_diag.len(), 2);
        assert!((super_diag[0] - 2.0).abs() < 1e-10);
        assert!((super_diag[1] - 6.0).abs() < 1e-10);

        // Sub-diagonal (k=-1)
        let sub_diag = diagonal_csr(&mat, -1);
        assert_eq!(sub_diag.len(), 2);
        assert!((sub_diag[0] - 4.0).abs() < 1e-10);
        assert!((sub_diag[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_triu_with_offset() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        // k=1: exclude main diagonal
        let upper = triu_csr(&mat, 1).unwrap();
        assert_eq!(upper.nnz(), 3); // 2,3,6
    }

    #[test]
    fn test_tril_with_offset() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        // k=-1: exclude main diagonal
        let lower = tril_csr(&mat, -1).unwrap();
        assert_eq!(lower.nnz(), 3); // 4,7,8
    }

    #[test]
    fn test_empty_matrix_operations() {
        assert!(vstack_csr::<f64>(&[]).is_err());
        assert!(hstack_csr::<f64>(&[]).is_err());
        assert!(block_diag_csr::<f64>(&[]).is_err());
    }
}
