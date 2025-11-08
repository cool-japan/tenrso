//! Unified sparse tensor operations interface
//!
//! This module provides a consistent API for sparse operations across different formats.
//!
//! # Operations
//!
//! - **SpMV**: Sparse Matrix-Vector multiplication
//! - **SpMM**: Sparse Matrix-Matrix multiplication (dense result)
//! - **SpSpMM**: Sparse-Sparse Matrix multiplication (sparse result)
//! - **Masked Operations**: Boolean mask-based operations
//! - **Element-wise**: Hadamard product, addition, subtraction
//! - **Reductions**: Sum, max, min along axes
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::ops::{SparseOps, SparseMatrixOps};
//! use tenrso_sparse::CsrMatrix;
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create a sparse matrix
//! let row_ptr = vec![0, 2, 4];
//! let col_indices = vec![0, 1, 1, 2];
//! let values = vec![1.0, 2.0, 3.0, 4.0];
//! let shape = (2, 3);
//!
//! let mat = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();
//!
//! // Sparse matrix-vector multiplication
//! let vec = array![1.0, 2.0, 3.0];
//! let result = mat.spmv(&vec.view()).unwrap();
//! assert_eq!(result.len(), 2);
//! ```

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;

/// Trait for sparse matrix operations
///
/// This trait defines the core operations that all sparse matrix formats should support.
pub trait SparseMatrixOps<T: Float> {
    /// Sparse matrix-vector multiplication: y = A * x
    ///
    /// # Complexity
    ///
    /// O(nnz) where nnz is the number of nonzeros
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match
    fn spmv(&self, x: &ArrayView1<T>) -> Result<Array1<T>>;

    /// Sparse matrix-matrix multiplication: C = A * B (dense result)
    ///
    /// # Complexity
    ///
    /// O(nnz * k) where k is the number of columns in B
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match
    fn spmm(&self, b: &ArrayView2<T>) -> Result<Array2<T>>;

    /// Get the number of rows
    fn nrows(&self) -> usize;

    /// Get the number of columns
    fn ncols(&self) -> usize;

    /// Get the number of nonzeros
    fn nnz(&self) -> usize;

    /// Get the density (nnz / total_elements)
    fn density(&self) -> f64 {
        let total = self.nrows() * self.ncols();
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }
}

/// Trait for sparse-sparse matrix operations
pub trait SparseSparseOps<T: Float> {
    /// Sparse-sparse matrix multiplication: C = A * B (sparse result)
    ///
    /// # Complexity
    ///
    /// O(m * avg_nnz_a * avg_nnz_b) where m is the number of rows in A
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match
    fn spspmm(&self, b: &Self) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for general sparse tensor operations
pub trait SparseOps<T: Float> {
    /// Get the shape of the tensor
    fn shape(&self) -> &[usize];

    /// Get the number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the number of nonzeros
    fn nnz(&self) -> usize;

    /// Get the density
    fn density(&self) -> f64;

    /// Convert to dense array
    fn to_dense(&self) -> Result<scirs2_core::ndarray_ext::ArrayD<T>>;
}

// Implement SparseMatrixOps for CsrMatrix
impl<T: Float> SparseMatrixOps<T> for CsrMatrix<T> {
    fn spmv(&self, x: &ArrayView1<T>) -> Result<Array1<T>> {
        Ok(CsrMatrix::spmv(self, x)?)
    }

    fn spmm(&self, b: &ArrayView2<T>) -> Result<Array2<T>> {
        Ok(CsrMatrix::spmm(self, b)?)
    }

    fn nrows(&self) -> usize {
        CsrMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        CsrMatrix::ncols(self)
    }

    fn nnz(&self) -> usize {
        CsrMatrix::nnz(self)
    }
}

// Implement SparseMatrixOps for CscMatrix
impl<T: Float> SparseMatrixOps<T> for CscMatrix<T> {
    fn spmv(&self, x: &ArrayView1<T>) -> Result<Array1<T>> {
        Ok(CscMatrix::matvec(self, x)?)
    }

    fn spmm(&self, b: &ArrayView2<T>) -> Result<Array2<T>> {
        Ok(CscMatrix::spmm(self, b)?)
    }

    fn nrows(&self) -> usize {
        CscMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        CscMatrix::ncols(self)
    }

    fn nnz(&self) -> usize {
        CscMatrix::nnz(self)
    }
}

// Implement SparseSparseOps for CsrMatrix
impl<T: Float> SparseSparseOps<T> for CsrMatrix<T> {
    fn spspmm(&self, b: &Self) -> Result<Self> {
        Ok(CsrMatrix::spspmm(self, b)?)
    }
}

// Implement SparseSparseOps for CscMatrix
impl<T: Float> SparseSparseOps<T> for CscMatrix<T> {
    fn spspmm(&self, b: &Self) -> Result<Self> {
        Ok(CscMatrix::spspmm(self, b)?)
    }
}

// Masked operations and sparse einsum traits are placeholders for M4 integration
// They will be implemented when integrating with tenrso-planner and tenrso-exec

/// Helper functions for sparse operations
/// Check if two sparse tensors have compatible shapes for element-wise operations
pub fn check_eltwise_compatible(shape_a: &[usize], shape_b: &[usize]) -> Result<()> {
    use crate::error::ShapeMismatchError;
    use crate::error::SparseError;

    if shape_a != shape_b {
        return Err(SparseError::ShapeMismatch(ShapeMismatchError::Tensor {
            expected: shape_a.to_vec(),
            got: shape_b.to_vec(),
        })
        .into());
    }
    Ok(())
}

/// Check if two matrices have compatible shapes for matrix multiplication
pub fn check_matmul_compatible(
    a_shape: (usize, usize),
    b_shape: (usize, usize),
) -> Result<(usize, usize)> {
    use crate::error::ShapeMismatchError;
    use crate::error::SparseError;

    let (m1, n1) = a_shape;
    let (m2, n2) = b_shape;

    if n1 != m2 {
        return Err(
            SparseError::ShapeMismatch(ShapeMismatchError::MatMul { m1, n1, m2, n2 }).into(),
        );
    }

    Ok((m1, n2))
}

/// Estimate output sparsity for sparse-sparse operations
///
/// This is a heuristic based on input densities.
///
/// # Arguments
///
/// - `nnz_a`: Number of nonzeros in A
/// - `nnz_b`: Number of nonzeros in B
/// - `m`: Number of rows in result
/// - `n`: Number of columns in result
///
/// # Returns
///
/// Estimated number of nonzeros in result
pub fn estimate_output_nnz(nnz_a: usize, nnz_b: usize, m: usize, n: usize) -> usize {
    // Heuristic: assume uniform distribution
    // Expected nnz in result ≈ min(m*n, sqrt(nnz_a * nnz_b))
    let max_nnz = m * n;
    let estimated = ((nnz_a as f64).sqrt() * (nnz_b as f64).sqrt()) as usize;
    estimated.min(max_nnz)
}

/// Sparse tensor addition (element-wise)
///
/// Computes C = α*A + β*B where A and B are sparse.
///
/// # Arguments
///
/// - `a`: First sparse matrix (CSR format)
/// - `b`: Second sparse matrix (CSR format)
/// - `alpha`: Scalar multiplier for A
/// - `beta`: Scalar multiplier for B
///
/// # Returns
///
/// Sparse matrix C = α*A + β*B in CSR format
///
/// # Examples
///
/// ```
/// use tenrso_sparse::ops::sparse_add_csr;
/// use tenrso_sparse::CsrMatrix;
///
/// let a = CsrMatrix::new(
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![1.0, 2.0, 3.0, 4.0],
///     (2, 2),
/// ).unwrap();
///
/// let b = CsrMatrix::new(
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![5.0, 6.0, 7.0, 8.0],
///     (2, 2),
/// ).unwrap();
///
/// let c = sparse_add_csr(&a, &b, 1.0, 1.0).unwrap();
/// assert_eq!(c.nnz(), 4);
/// ```
pub fn sparse_add_csr<T: Float>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    alpha: T,
    beta: T,
) -> Result<CsrMatrix<T>> {
    check_eltwise_compatible(&[a.nrows(), a.ncols()], &[b.nrows(), b.ncols()])?;

    let m = a.nrows();
    let n = a.ncols();

    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);

    for i in 0..m {
        // Get rows from both matrices
        let a_start = a.row_ptr()[i];
        let a_end = a.row_ptr()[i + 1];
        let b_start = b.row_ptr()[i];
        let b_end = b.row_ptr()[i + 1];

        let a_cols = &a.col_indices()[a_start..a_end];
        let a_vals = &a.values()[a_start..a_end];
        let b_cols = &b.col_indices()[b_start..b_end];
        let b_vals = &b.values()[b_start..b_end];

        // Merge the two sorted arrays
        let mut ai = 0;
        let mut bi = 0;

        while ai < a_cols.len() || bi < b_cols.len() {
            let val = if ai < a_cols.len() && (bi >= b_cols.len() || a_cols[ai] < b_cols[bi]) {
                // Only in A
                let v = alpha * a_vals[ai];
                let col = a_cols[ai];
                ai += 1;
                (col, v)
            } else if bi < b_cols.len() && (ai >= a_cols.len() || b_cols[bi] < a_cols[ai]) {
                // Only in B
                let v = beta * b_vals[bi];
                let col = b_cols[bi];
                bi += 1;
                (col, v)
            } else {
                // In both
                let v = alpha * a_vals[ai] + beta * b_vals[bi];
                let col = a_cols[ai];
                ai += 1;
                bi += 1;
                (col, v)
            };

            // Only add if nonzero (using zero threshold)
            let zero_threshold = T::zero() + T::zero(); // Essentially zero
            if val.1.abs() > zero_threshold {
                col_indices.push(val.0);
                values.push(val.1);
            }
        }

        row_ptr.push(col_indices.len());
    }

    Ok(CsrMatrix::new(row_ptr, col_indices, values, (m, n))?)
}

/// Sparse matrix scalar multiplication
///
/// Computes C = α * A where A is sparse.
///
/// # Arguments
///
/// - `a`: Sparse matrix (CSR format)
/// - `alpha`: Scalar multiplier
///
/// # Returns
///
/// Sparse matrix C = α * A in CSR format
///
/// # Examples
///
/// ```
/// use tenrso_sparse::ops::sparse_scale_csr;
/// use tenrso_sparse::CsrMatrix;
///
/// let a = CsrMatrix::new(
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![1.0, 2.0, 3.0, 4.0],
///     (2, 2),
/// ).unwrap();
///
/// let c = sparse_scale_csr(&a, 2.0).unwrap();
/// assert_eq!(c.nnz(), 4);
/// // Values should be [2.0, 4.0, 6.0, 8.0]
/// ```
pub fn sparse_scale_csr<T: Float>(a: &CsrMatrix<T>, alpha: T) -> Result<CsrMatrix<T>> {
    let scaled_values: Vec<T> = a.values().iter().map(|&v| alpha * v).collect();

    Ok(CsrMatrix::new(
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        scaled_values,
        (a.nrows(), a.ncols()),
    )?)
}

/// Sparse matrix transpose (CSR to CSC conversion)
///
/// Computes A^T where A is sparse CSR.
///
/// # Arguments
///
/// - `a`: Sparse matrix (CSR format)
///
/// # Returns
///
/// Transposed matrix in CSC format
///
/// # Complexity
///
/// O(nnz) time and space
///
/// # Examples
///
/// ```
/// use tenrso_sparse::ops::sparse_transpose_csr;
/// use tenrso_sparse::CsrMatrix;
///
/// let a = CsrMatrix::new(
///     vec![0, 2, 3],
///     vec![0, 1, 2],
///     vec![1.0, 2.0, 3.0],
///     (2, 3),
/// ).unwrap();
///
/// let at = sparse_transpose_csr(&a).unwrap();
/// assert_eq!(at.nrows(), 3);  // Transposed dimensions
/// assert_eq!(at.ncols(), 2);
/// assert_eq!(at.nnz(), 3);
/// ```
pub fn sparse_transpose_csr<T: Float>(a: &CsrMatrix<T>) -> Result<crate::csc::CscMatrix<T>> {
    Ok(a.to_csc())
}

/// Sparse matrix element-wise absolute value
///
/// Computes C = abs(A) where A is sparse.
///
/// # Arguments
///
/// - `a`: Sparse matrix (CSR format)
///
/// # Returns
///
/// Sparse matrix C with absolute values in CSR format
///
/// # Examples
///
/// ```
/// use tenrso_sparse::ops::sparse_abs_csr;
/// use tenrso_sparse::CsrMatrix;
///
/// let a = CsrMatrix::new(
///     vec![0, 2],
///     vec![0, 1],
///     vec![-1.0, 2.0],
///     (1, 2),
/// ).unwrap();
///
/// let c = sparse_abs_csr(&a).unwrap();
/// assert_eq!(c.values()[0], 1.0);  // abs(-1.0)
/// assert_eq!(c.values()[1], 2.0);  // abs(2.0)
/// ```
pub fn sparse_abs_csr<T: Float>(a: &CsrMatrix<T>) -> Result<CsrMatrix<T>> {
    let abs_values: Vec<T> = a.values().iter().map(|&v| v.abs()).collect();

    Ok(CsrMatrix::new(
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        abs_values,
        (a.nrows(), a.ncols()),
    )?)
}

/// Sparse matrix element-wise square
///
/// Computes C = A .* A (element-wise) where A is sparse.
///
/// # Arguments
///
/// - `a`: Sparse matrix (CSR format)
///
/// # Returns
///
/// Sparse matrix C with squared values in CSR format
///
/// # Examples
///
/// ```
/// use tenrso_sparse::ops::sparse_square_csr;
/// use tenrso_sparse::CsrMatrix;
///
/// let a = CsrMatrix::new(
///     vec![0, 2],
///     vec![0, 1],
///     vec![2.0, 3.0],
///     (1, 2),
/// ).unwrap();
///
/// let c = sparse_square_csr(&a).unwrap();
/// assert_eq!(c.values()[0], 4.0);  // 2^2
/// assert_eq!(c.values()[1], 9.0);  // 3^2
/// ```
pub fn sparse_square_csr<T: Float>(a: &CsrMatrix<T>) -> Result<CsrMatrix<T>> {
    let squared_values: Vec<T> = a.values().iter().map(|&v| v * v).collect();

    Ok(CsrMatrix::new(
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        squared_values,
        (a.nrows(), a.ncols()),
    )?)
}

/// Count nonzeros per row in CSR matrix
///
/// Returns a vector where element i contains the number of nonzeros in row i.
///
/// # Arguments
///
/// - `a`: Sparse matrix (CSR format)
///
/// # Returns
///
/// Vector of nonzero counts per row
///
/// # Examples
///
/// ```
/// use tenrso_sparse::ops::nnz_per_row_csr;
/// use tenrso_sparse::CsrMatrix;
///
/// let a = CsrMatrix::new(
///     vec![0, 2, 3, 5],
///     vec![0, 1, 2, 0, 1],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0],
///     (3, 3),
/// ).unwrap();
///
/// let nnz_counts = nnz_per_row_csr(&a);
/// assert_eq!(nnz_counts, vec![2, 1, 2]);
/// ```
pub fn nnz_per_row_csr<T: Float>(a: &CsrMatrix<T>) -> Vec<usize> {
    let mut counts = Vec::with_capacity(a.nrows());
    let row_ptr = a.row_ptr();

    for i in 0..a.nrows() {
        counts.push(row_ptr[i + 1] - row_ptr[i]);
    }

    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_spmv_via_trait() {
        let mat = CsrMatrix::new(
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            (2, 3),
        )
        .unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = mat.spmv(&x.view()).unwrap();

        assert_eq!(y.len(), 2);
        assert!((y[0] - 5.0).abs() < 1e-10); // 1*1 + 2*2 = 5
        assert!((y[1] - 18.0).abs() < 1e-10); // 3*2 + 4*3 = 18
    }

    #[test]
    fn test_sparse_matrix_ops_csr() {
        let mat = CsrMatrix::new(vec![0, 2], vec![0, 1], vec![1.0, 2.0], (1, 3)).unwrap();

        assert_eq!(mat.nrows(), 1);
        assert_eq!(mat.ncols(), 3);
        assert_eq!(mat.nnz(), 2);
        assert!((mat.density() - (2.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_check_matmul_compatible() {
        assert!(check_matmul_compatible((3, 4), (4, 5)).is_ok());
        assert!(check_matmul_compatible((3, 4), (5, 6)).is_err());

        let result = check_matmul_compatible((3, 4), (4, 5)).unwrap();
        assert_eq!(result, (3, 5));
    }

    #[test]
    fn test_check_eltwise_compatible() {
        assert!(check_eltwise_compatible(&[3, 4], &[3, 4]).is_ok());
        assert!(check_eltwise_compatible(&[3, 4], &[4, 3]).is_err());
    }

    #[test]
    fn test_estimate_output_nnz() {
        let nnz = estimate_output_nnz(100, 100, 1000, 1000);
        assert!(nnz > 0);
        assert!(nnz <= 1_000_000);
    }

    #[test]
    fn test_sparse_add_csr() {
        let a = CsrMatrix::new(
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            (2, 2),
        )
        .unwrap();

        let b = CsrMatrix::new(
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![5.0, 6.0, 7.0, 8.0],
            (2, 2),
        )
        .unwrap();

        let c = sparse_add_csr(&a, &b, 1.0, 1.0).unwrap();

        assert_eq!(c.nnz(), 4);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);

        // Check values: C[0,0] = 1+5=6, C[0,1] = 2+6=8, C[1,0] = 3+7=10, C[1,1] = 4+8=12
        let vals = c.values();
        assert!((vals[0] - 6.0).abs() < 1e-10);
        assert!((vals[1] - 8.0).abs() < 1e-10);
        assert!((vals[2] - 10.0).abs() < 1e-10);
        assert!((vals[3] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_add_csr_different_sparsity() {
        let a = CsrMatrix::new(vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0], (2, 2)).unwrap();

        let b = CsrMatrix::new(vec![0, 1, 2], vec![1, 0], vec![3.0, 4.0], (2, 2)).unwrap();

        let c = sparse_add_csr(&a, &b, 1.0, 1.0).unwrap();

        // A has (0,0)=1, (1,1)=2
        // B has (0,1)=3, (1,0)=4
        // C should have all 4: (0,0)=1, (0,1)=3, (1,0)=4, (1,1)=2
        assert_eq!(c.nnz(), 4);
    }

    #[test]
    fn test_sparse_scale_csr() {
        let a = CsrMatrix::new(
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            (2, 2),
        )
        .unwrap();

        let c = sparse_scale_csr(&a, 2.0).unwrap();

        assert_eq!(c.nnz(), 4);
        assert_eq!(c.values()[0], 2.0);
        assert_eq!(c.values()[1], 4.0);
        assert_eq!(c.values()[2], 6.0);
        assert_eq!(c.values()[3], 8.0);
    }

    #[test]
    fn test_sparse_transpose_csr() {
        let a = CsrMatrix::new(vec![0, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0], (2, 3)).unwrap();

        let at = sparse_transpose_csr(&a).unwrap();

        assert_eq!(at.nrows(), 3);
        assert_eq!(at.ncols(), 2);
        assert_eq!(at.nnz(), 3);
    }

    #[test]
    fn test_sparse_abs_csr() {
        let a = CsrMatrix::new(vec![0, 3], vec![0, 1, 2], vec![-1.0, 2.0, -3.0], (1, 3)).unwrap();

        let c = sparse_abs_csr(&a).unwrap();

        assert_eq!(c.values()[0], 1.0);
        assert_eq!(c.values()[1], 2.0);
        assert_eq!(c.values()[2], 3.0);
    }

    #[test]
    fn test_sparse_square_csr() {
        let a = CsrMatrix::new(vec![0, 2], vec![0, 1], vec![2.0, 3.0], (1, 2)).unwrap();

        let c = sparse_square_csr(&a).unwrap();

        assert_eq!(c.values()[0], 4.0);
        assert_eq!(c.values()[1], 9.0);
    }

    #[test]
    fn test_nnz_per_row_csr() {
        let a = CsrMatrix::new(
            vec![0, 2, 3, 5],
            vec![0, 1, 2, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            (3, 3),
        )
        .unwrap();

        let counts = nnz_per_row_csr(&a);
        assert_eq!(counts, vec![2, 1, 2]);
    }
}
