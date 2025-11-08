//! CSC (Compressed Sparse Column) format for 2D matrices
//!
//! CSC is the column-major dual of CSR, optimized for column-wise operations.
//!
//! # Format
//!
//! For an m×n sparse matrix with nnz non-zeros:
//! - `col_ptr`: `Vec<usize>` of length n+1 - col_ptr\[j\] points to start of column j
//! - `row_indices`: `Vec<usize>` of length nnz - row index for each non-zero
//! - `values`: `Vec<T>` of length nnz - the non-zero values
//! - `shape`: (m, n) - dimensions of the matrix
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::csc::CscMatrix;
//!
//! // Create a 3×3 sparse matrix:
//! // [1.0  0   4.0]
//! // [0    3.0 0  ]
//! // [2.0  0   5.0]
//!
//! let col_ptr = vec![0, 2, 3, 5];  // Cumulative column starts
//! let row_indices = vec![0, 2, 1, 0, 2];
//! let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 3);
//!
//! let csc = CscMatrix::new(col_ptr, row_indices, values, shape).unwrap();
//! assert_eq!(csc.nnz(), 5);
//! ```
//!
//! # SciRS2 Integration
//!
//! All operations use `scirs2_core` types. Direct use of `ndarray` is forbidden.

use crate::coo::{CooError, CooTensor};
use crate::csr::CsrMatrix;
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CscError {
    #[error("Invalid column pointers: length {len} for {ncols} columns (expected {expected})")]
    InvalidColPtr {
        len: usize,
        ncols: usize,
        expected: usize,
    },

    #[error("Column pointer not sorted at index {idx}: {curr} > {next}")]
    ColPtrNotSorted {
        idx: usize,
        curr: usize,
        next: usize,
    },

    #[error("Length mismatch: {row_indices} row_indices but {values} values")]
    LengthMismatch { row_indices: usize, values: usize },

    #[error("Row index out of bounds: {row_idx} >= {nrows}")]
    RowIndexOutOfBounds { row_idx: usize, nrows: usize },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Matrix shape mismatch: cannot multiply {m1}×{n1} by {m2}×{n2}")]
    MatrixShapeMismatch {
        m1: usize,
        n1: usize,
        m2: usize,
        n2: usize,
    },

    #[error("COO conversion error: {0}")]
    CooError(#[from] CooError),
}

/// CSC (Compressed Sparse Column) matrix
///
/// Optimized for column-wise operations and transpose operations.
#[derive(Debug, Clone)]
pub struct CscMatrix<T> {
    /// Column pointers: col_ptr[j] = start index of column j in row_indices/values
    /// Length: ncols + 1, with col_ptr[ncols] = nnz
    col_ptr: Vec<usize>,

    /// Row indices for each non-zero element
    row_indices: Vec<usize>,

    /// Values of non-zero elements
    values: Vec<T>,

    /// Shape: (nrows, ncols)
    shape: (usize, usize),
}

impl<T: Clone> CscMatrix<T> {
    /// Create a new CSC matrix
    ///
    /// # Arguments
    ///
    /// * `col_ptr` - Column pointers (length ncols+1)
    /// * `row_indices` - Row indices for each non-zero
    /// * `values` - Values for each non-zero
    /// * `shape` - (nrows, ncols)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - col_ptr length is incorrect
    /// - row_indices and values have different lengths
    /// - col_ptr is not monotonically increasing
    /// - any row index is out of bounds
    pub fn new(
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
        shape: (usize, usize),
    ) -> Result<Self, CscError> {
        let (nrows, ncols) = shape;

        // Validate shape
        if nrows == 0 || ncols == 0 {
            return Err(CscError::InvalidShape(
                "Shape cannot have zeros".to_string(),
            ));
        }

        // Validate col_ptr length
        if col_ptr.len() != ncols + 1 {
            return Err(CscError::InvalidColPtr {
                len: col_ptr.len(),
                ncols,
                expected: ncols + 1,
            });
        }

        // Validate row_indices and values length
        if row_indices.len() != values.len() {
            return Err(CscError::LengthMismatch {
                row_indices: row_indices.len(),
                values: values.len(),
            });
        }

        // Validate col_ptr is monotonically increasing
        for j in 0..ncols {
            if col_ptr[j] > col_ptr[j + 1] {
                return Err(CscError::ColPtrNotSorted {
                    idx: j,
                    curr: col_ptr[j],
                    next: col_ptr[j + 1],
                });
            }
        }

        // Validate final col_ptr matches nnz
        let nnz = row_indices.len();
        if col_ptr[ncols] != nnz {
            return Err(CscError::InvalidColPtr {
                len: col_ptr[ncols],
                ncols,
                expected: nnz,
            });
        }

        // Validate row indices
        for &row_idx in &row_indices {
            if row_idx >= nrows {
                return Err(CscError::RowIndexOutOfBounds { row_idx, nrows });
            }
        }

        Ok(Self {
            col_ptr,
            row_indices,
            values,
            shape,
        })
    }

    /// Create an empty CSC matrix with given shape
    pub fn zeros(shape: (usize, usize)) -> Result<Self, CscError> {
        let (nrows, ncols) = shape;
        if nrows == 0 || ncols == 0 {
            return Err(CscError::InvalidShape(
                "Shape cannot have zeros".to_string(),
            ));
        }

        Ok(Self {
            col_ptr: vec![0; ncols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
            shape,
        })
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Shape of the matrix (nrows, ncols)
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Number of rows
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get column pointers
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Get row indices
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get values
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Compute density (nnz / total_elements)
    pub fn density(&self) -> f64 {
        let total = self.nrows() * self.ncols();
        self.nnz() as f64 / total as f64
    }

    /// Get a column as (row_indices, values) slices
    pub fn column(&self, j: usize) -> Option<(&[usize], &[T])> {
        if j >= self.ncols() {
            return None;
        }

        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];

        Some((&self.row_indices[start..end], &self.values[start..end]))
    }
}

impl<T: Float> CscMatrix<T> {
    /// Convert from COO format
    ///
    /// The input COO tensor must be 2-dimensional.
    pub fn from_coo(coo: &CooTensor<T>) -> Result<Self, CscError> {
        if coo.rank() != 2 {
            return Err(CscError::InvalidShape(format!(
                "COO tensor must be 2D, got {}D",
                coo.rank()
            )));
        }

        let nrows = coo.shape()[0];
        let ncols = coo.shape()[1];
        let nnz = coo.nnz();

        // Sort COO by column then row
        let mut coo_sorted = coo.clone();
        coo_sorted.sort();

        // Need to re-sort by (col, row) order for CSC
        let mut triplets: Vec<(usize, usize, T)> = coo_sorted
            .indices()
            .iter()
            .zip(coo_sorted.values())
            .map(|(idx, &val)| (idx[0], idx[1], val))
            .collect();

        triplets.sort_by_key(|&(row, col, _)| (col, row));

        // Build CSC
        let mut col_ptr = vec![0; ncols + 1];
        let mut row_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Count elements per column
        for &(_row, col, _val) in &triplets {
            col_ptr[col + 1] += 1;
        }

        // Cumulative sum to get column starts
        for j in 0..ncols {
            col_ptr[j + 1] += col_ptr[j];
        }

        // Fill row_indices and values
        for (row, _col, val) in triplets {
            row_indices.push(row);
            values.push(val);
        }

        Self::new(col_ptr, row_indices, values, (nrows, ncols))
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> CooTensor<T> {
        let mut indices = Vec::with_capacity(self.nnz());
        let values = self.values.clone();

        for col in 0..self.ncols() {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            for row_idx in start..end {
                let row = self.row_indices[row_idx];
                indices.push(vec![row, col]);
            }
        }

        let shape = vec![self.nrows(), self.ncols()];
        CooTensor::new(indices, values, shape).unwrap()
    }

    /// Convert to CSR format (transpose storage)
    pub fn to_csr(&self) -> CsrMatrix<T> {
        // CSC is essentially the transpose of CSR
        // So CSC(A) -> CSR(A^T)
        let coo = self.to_coo();
        CsrMatrix::from_coo(&coo).unwrap()
    }

    /// Convert from CSR format (transpose storage)
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        // CSR(A) -> CSC(A^T)
        let coo = csr.to_coo();
        Self::from_coo(&coo).unwrap()
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Result<DenseND<T>> {
        let (nrows, ncols) = self.shape;
        let mut data = vec![T::zero(); nrows * ncols];

        for col in 0..ncols {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            for idx in start..end {
                let row = self.row_indices[idx];
                let value = self.values[idx];
                data[row * ncols + col] = value;
            }
        }

        DenseND::from_vec(data, &[nrows, ncols])
    }

    /// Create CSC from dense matrix
    ///
    /// Only stores elements where |value| > threshold.
    pub fn from_dense(dense: &DenseND<T>, threshold: T) -> Result<Self, CscError> {
        if dense.rank() != 2 {
            return Err(CscError::InvalidShape(format!(
                "Dense tensor must be 2D, got {}D",
                dense.rank()
            )));
        }

        // First convert to COO, then to CSC
        let coo = CooTensor::from_dense(dense, threshold);
        Self::from_coo(&coo)
    }

    /// Matrix-Vector product with column-major access: y = A * x
    ///
    /// Uses column-wise accumulation which is cache-friendly for CSC.
    ///
    /// # Complexity
    ///
    /// O(nnz) - linear in number of non-zeros
    pub fn matvec(&self, x: &ArrayView1<T>) -> Result<Array1<T>, CscError> {
        if x.len() != self.ncols() {
            return Err(CscError::InvalidShape(format!(
                "Vector length {} doesn't match matrix columns {}",
                x.len(),
                self.ncols()
            )));
        }

        let mut y = Array1::<T>::zeros(self.nrows());

        // Column-wise accumulation: y += A[:, j] * x[j]
        for col in 0..self.ncols() {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            let x_col = x[col];

            for idx in start..end {
                let row = self.row_indices[idx];
                let value = self.values[idx];
                y[row] = y[row] + value * x_col;
            }
        }

        Ok(y)
    }

    /// Matrix-Matrix product with column-major access: C = A * B
    ///
    /// Uses column-wise computation which is cache-friendly for CSC format.
    ///
    /// # Arguments
    ///
    /// * `b` - Dense matrix of shape (ncols, k)
    ///
    /// # Returns
    ///
    /// Dense matrix C of shape (nrows, k) where C = A * B
    ///
    /// # Errors
    ///
    /// Returns error if A.ncols != B.nrows
    ///
    /// # Complexity
    ///
    /// O(nnz * k) where k is the number of columns in B
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::csc::CscMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Sparse matrix A (CSC): [1 0 4]
    /// //                         [0 3 0]
    /// //                         [2 0 5]
    /// let col_ptr = vec![0, 2, 3, 5];
    /// let row_indices = vec![0, 2, 1, 0, 2];
    /// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let csc = CscMatrix::new(col_ptr, row_indices, values, (3, 3)).unwrap();
    ///
    /// // Dense matrix B: [1 2]
    /// //                 [3 4]
    /// //                 [5 6]
    /// let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    ///
    /// let c = csc.spmm(&b.view()).unwrap();
    /// assert_eq!(c[[0, 0]], 21.0);  // 1*1 + 4*5
    /// assert_eq!(c[[0, 1]], 26.0);  // 1*2 + 4*6
    /// ```
    pub fn spmm(&self, b: &ArrayView2<T>) -> Result<Array2<T>, CscError> {
        let (b_rows, b_cols) = (b.nrows(), b.ncols());

        // Validate dimensions: A is (m, n), B is (n, k) -> C is (m, k)
        if self.ncols() != b_rows {
            return Err(CscError::MatrixShapeMismatch {
                m1: self.nrows(),
                n1: self.ncols(),
                m2: b_rows,
                n2: b_cols,
            });
        }

        // Allocate result matrix
        let mut c = Array2::<T>::zeros((self.nrows(), b_cols));

        // Compute C = A * B using column-wise accumulation
        // For each column j in A, for each output column k:
        //   C[:, k] += A[:, j] * B[j, k]
        for j in 0..self.ncols() {
            let col_start = self.col_ptr[j];
            let col_end = self.col_ptr[j + 1];

            for k in 0..b_cols {
                let b_jk = b[[j, k]];

                // Accumulate A[:, j] * B[j, k] into C[:, k]
                for idx in col_start..col_end {
                    let row = self.row_indices[idx];
                    let a_val = self.values[idx];
                    c[[row, k]] = c[[row, k]] + a_val * b_jk;
                }
            }
        }

        Ok(c)
    }

    /// Sparse matrix-matrix multiply: C = A * B (sparse × sparse → sparse)
    ///
    /// Multiplies two sparse matrices and produces a sparse result.
    /// More efficient than `spmm` when the result is expected to be sparse.
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand sparse matrix (CSC format)
    ///
    /// # Returns
    ///
    /// A new CSC matrix representing the product
    ///
    /// # Complexity
    ///
    /// Time: O(m × nnz_per_col_A × nnz_per_col_B)
    /// Space: O(nnz_result)
    ///
    /// Uses column-wise accumulation for efficient sparse result construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::csc::CscMatrix;
    /// use tenrso_core::DenseND;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Create two sparse matrices
    /// let a_dense = DenseND::from_array(array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]].into_dyn());
    /// let b_dense = DenseND::from_array(array![[1.0, 0.0], [0.0, 0.0], [0.0, 4.0]].into_dyn());
    ///
    /// let a = CscMatrix::from_dense(&a_dense, 1e-10).unwrap();
    /// let b = CscMatrix::from_dense(&b_dense, 1e-10).unwrap();
    ///
    /// let c = a.spspmm(&b).unwrap();
    ///
    /// assert_eq!(c.shape(), (2, 2));
    /// // Result: [[1.0, 8.0], [0.0, 0.0]]
    /// ```
    pub fn spspmm(&self, b: &CscMatrix<T>) -> Result<CscMatrix<T>, CscError> {
        // Validate dimensions: A is (m, n), B is (n, k) -> C is (m, k)
        if self.ncols() != b.nrows() {
            return Err(CscError::MatrixShapeMismatch {
                m1: self.nrows(),
                n1: self.ncols(),
                m2: b.nrows(),
                n2: b.ncols(),
            });
        }

        let m = self.nrows();
        let k = b.ncols();
        use std::collections::HashMap;

        let mut result_col_ptr = vec![0];
        let mut result_row_indices = Vec::new();
        let mut result_values = Vec::new();

        // For each column j in B (result will have k columns)
        for j in 0..k {
            let mut col_map: HashMap<usize, T> = HashMap::new();

            // For each non-zero B[i, j] in column j of B
            for b_idx in b.col_ptr[j]..b.col_ptr[j + 1] {
                let i = b.row_indices[b_idx];
                let b_val = b.values[b_idx];

                // For each non-zero A[row, i] in column i of A
                for a_idx in self.col_ptr[i]..self.col_ptr[i + 1] {
                    let row = self.row_indices[a_idx];
                    let a_val = self.values[a_idx];

                    // Accumulate A[row, i] * B[i, j] into result[row, j]
                    let entry = col_map.entry(row).or_insert(T::zero());
                    *entry = *entry + a_val * b_val;
                }
            }

            // Convert col_map to sorted vectors (sort by row index)
            let mut col_entries: Vec<_> = col_map.into_iter().collect();
            col_entries.sort_by_key(|(row, _)| *row);

            // Add non-zero entries to result
            for (row, val) in col_entries {
                if val != T::zero() {
                    result_row_indices.push(row);
                    result_values.push(val);
                }
            }

            result_col_ptr.push(result_row_indices.len());
        }

        CscMatrix::new(result_col_ptr, result_row_indices, result_values, (m, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_creation() {
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let csc = CscMatrix::new(col_ptr, row_indices, values, shape).unwrap();
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
    }

    #[test]
    fn test_csc_zeros() {
        let csc = CscMatrix::<f64>::zeros((5, 4)).unwrap();
        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.shape(), (5, 4));
    }

    #[test]
    fn test_csc_column_access() {
        // Matrix: [1 0 4]
        //         [0 3 0]
        //         [2 0 5]
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let csc = CscMatrix::new(col_ptr, row_indices, values, shape).unwrap();

        let (rows, vals) = csc.column(0).unwrap();
        assert_eq!(rows, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);

        let (rows, vals) = csc.column(1).unwrap();
        assert_eq!(rows, &[1]);
        assert_eq!(vals, &[3.0]);

        let (rows, vals) = csc.column(2).unwrap();
        assert_eq!(rows, &[0, 2]);
        assert_eq!(vals, &[4.0, 5.0]);
    }

    #[test]
    fn test_csc_from_coo() {
        let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![2, 2]];
        let values = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let shape = vec![3, 3];

        let coo = CooTensor::new(indices, values, shape).unwrap();
        let csc = CscMatrix::from_coo(&coo).unwrap();

        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), (3, 3));
    }

    #[test]
    fn test_csc_to_coo() {
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let csc = CscMatrix::new(col_ptr, row_indices, values, shape).unwrap();
        let coo = csc.to_coo();

        assert_eq!(coo.nnz(), 5);
        assert_eq!(coo.shape(), &[3, 3]);
    }

    #[test]
    fn test_csc_to_dense() {
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let csc = CscMatrix::new(col_ptr, row_indices, values, shape).unwrap();
        let dense = csc.to_dense().unwrap();

        assert_eq!(dense.shape(), &[3, 3]);
        let view = dense.view();
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 2]], 4.0);
        assert_eq!(view[[1, 1]], 3.0);
        assert_eq!(view[[2, 0]], 2.0);
        assert_eq!(view[[2, 2]], 5.0);
    }

    #[test]
    fn test_csc_density() {
        let col_ptr = vec![0, 1, 2];
        let row_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let shape = (10, 2);

        let csc = CscMatrix::new(col_ptr, row_indices, values, shape).unwrap();
        assert_eq!(csc.density(), 0.1); // 2/20
    }

    #[test]
    fn test_csc_matvec() {
        use scirs2_core::ndarray_ext::array;

        // Matrix: [1 0 4]
        //         [0 3 0]
        //         [2 0 5]
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csc = CscMatrix::new(col_ptr, row_indices, values, (3, 3)).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = csc.matvec(&x.view()).unwrap();

        // y[0] = 1*1 + 4*3 = 13
        // y[1] = 3*2 = 6
        // y[2] = 2*1 + 5*3 = 17
        assert_eq!(y[0], 13.0);
        assert_eq!(y[1], 6.0);
        assert_eq!(y[2], 17.0);
    }

    #[test]
    fn test_csc_spmm_basic() {
        use scirs2_core::ndarray_ext::array;

        // Sparse matrix A (CSC): [1 0 4]
        //                         [0 3 0]
        //                         [2 0 5]
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csc = CscMatrix::new(col_ptr, row_indices, values, (3, 3)).unwrap();

        // Dense matrix B: [1 2]
        //                 [3 4]
        //                 [5 6]
        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = csc.spmm(&b.view()).unwrap();

        // Verify dimensions
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 2);

        // C = A * B
        // C[0,0] = 1*1 + 4*5 = 21
        // C[0,1] = 1*2 + 4*6 = 26
        // C[1,0] = 3*3 = 9
        // C[1,1] = 3*4 = 12
        // C[2,0] = 2*1 + 5*5 = 27
        // C[2,1] = 2*2 + 5*6 = 34
        assert_eq!(c[[0, 0]], 21.0);
        assert_eq!(c[[0, 1]], 26.0);
        assert_eq!(c[[1, 0]], 9.0);
        assert_eq!(c[[1, 1]], 12.0);
        assert_eq!(c[[2, 0]], 27.0);
        assert_eq!(c[[2, 1]], 34.0);
    }

    #[test]
    fn test_csc_spmm_identity() {
        use scirs2_core::ndarray_ext::array;

        // Identity matrix in CSC format
        let col_ptr = vec![0, 1, 2, 3];
        let row_indices = vec![0, 1, 2];
        let values = vec![1.0, 1.0, 1.0];
        let csc = CscMatrix::new(col_ptr, row_indices, values, (3, 3)).unwrap();

        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = csc.spmm(&b.view()).unwrap();

        // Should return B itself
        assert_eq!(c[[0, 0]], 1.0);
        assert_eq!(c[[0, 1]], 2.0);
        assert_eq!(c[[1, 0]], 3.0);
        assert_eq!(c[[1, 1]], 4.0);
        assert_eq!(c[[2, 0]], 5.0);
        assert_eq!(c[[2, 1]], 6.0);
    }

    #[test]
    fn test_csc_spmm_single_column() {
        use scirs2_core::ndarray_ext::array;

        // Test SpMM with single column (should match matvec)
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csc = CscMatrix::new(col_ptr, row_indices, values, (3, 3)).unwrap();

        let b = array![[1.0], [2.0], [3.0]];
        let c = csc.spmm(&b.view()).unwrap();

        // Compare with matvec result
        let x = array![1.0, 2.0, 3.0];
        let y = csc.matvec(&x.view()).unwrap();

        assert_eq!(c[[0, 0]], y[0]);
        assert_eq!(c[[1, 0]], y[1]);
        assert_eq!(c[[2, 0]], y[2]);
    }

    #[test]
    fn test_csc_spmm_empty_columns() {
        use scirs2_core::ndarray_ext::array;

        // Matrix with empty column: [1 0 4]
        //                            [2 0 5]
        let col_ptr = vec![0, 2, 2, 4];
        let row_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 4.0, 5.0];
        let csc = CscMatrix::new(col_ptr, row_indices, values, (2, 3)).unwrap();

        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = csc.spmm(&b.view()).unwrap();

        // First row: 1*1 + 4*5 = 21, 1*2 + 4*6 = 26
        assert_eq!(c[[0, 0]], 21.0);
        assert_eq!(c[[0, 1]], 26.0);

        // Second row: 2*1 + 5*5 = 27, 2*2 + 5*6 = 34
        assert_eq!(c[[1, 0]], 27.0);
        assert_eq!(c[[1, 1]], 34.0);
    }

    #[test]
    fn test_csc_spmm_shape_mismatch() {
        use scirs2_core::ndarray_ext::array;

        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csc = CscMatrix::new(col_ptr, row_indices, values, (3, 3)).unwrap();

        // Wrong size matrix: should be 3×k but is 2×2
        let b = array![[1.0, 2.0], [3.0, 4.0]];
        let result = csc.spmm(&b.view());

        assert!(result.is_err());
    }

    #[test]
    fn test_csc_spspmm_basic() {
        use scirs2_core::ndarray_ext::array;

        // A = [[1, 0, 2],    B = [[1, 0],
        //      [0, 3, 0]]         [0, 0],
        //                         [0, 4]]
        let a_array = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let b_array = array![[1.0, 0.0], [0.0, 0.0], [0.0, 4.0]];

        let a_dense = DenseND::from_array(a_array.into_dyn());
        let b_dense = DenseND::from_array(b_array.into_dyn());

        let a = CscMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CscMatrix::from_dense(&b_dense, 1e-10).unwrap();

        let c = a.spspmm(&b).unwrap();

        assert_eq!(c.shape(), (2, 2));

        // Verify result: C = A * B
        // C[0,0] = 1*1 + 0*0 + 2*0 = 1
        // C[0,1] = 1*0 + 0*0 + 2*4 = 8
        // C[1,0] = 0*1 + 3*0 + 0*0 = 0
        // C[1,1] = 0*0 + 3*0 + 0*4 = 0
        let c_dense = c.to_dense().unwrap();
        assert!((c_dense.as_array()[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((c_dense.as_array()[[0, 1]] - 8.0).abs() < 1e-10);
        assert!((c_dense.as_array()[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((c_dense.as_array()[[1, 1]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_csc_spspmm_identity() {
        use scirs2_core::ndarray_ext::array;

        // A = identity, B = sparse matrix
        let a_array = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b_array = array![[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]];

        let a_dense = DenseND::from_array(a_array.into_dyn());
        let b_dense = DenseND::from_array(b_array.into_dyn());

        let a = CscMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CscMatrix::from_dense(&b_dense, 1e-10).unwrap();

        let c = a.spspmm(&b).unwrap();

        // Result should be B
        let c_dense = c.to_dense().unwrap();
        let b_check = b.to_dense().unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert!((c_dense.as_array()[[i, j]] - b_check.as_array()[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_csc_spspmm_zeros() {
        // Test with all-zero matrices
        let a = CscMatrix::<f64>::zeros((3, 3)).unwrap();
        let b = CscMatrix::<f64>::zeros((3, 2)).unwrap();

        let c = a.spspmm(&b).unwrap();

        assert_eq!(c.nnz(), 0);
        assert_eq!(c.shape(), (3, 2));
    }

    #[test]
    fn test_csc_spspmm_correctness_vs_dense() {
        use scirs2_core::ndarray_ext::array;

        // Test that sparse-sparse multiply matches dense multiply
        let a_array = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 6.0, 0.0]
        ];
        let b_array = array![
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]
        ];

        let a_dense = DenseND::from_array(a_array.clone().into_dyn());
        let b_dense = DenseND::from_array(b_array.clone().into_dyn());

        let a = CscMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CscMatrix::from_dense(&b_dense, 1e-10).unwrap();

        let c_sparse = a.spspmm(&b).unwrap();
        let c_dense_expected: scirs2_core::ndarray_ext::Array2<f64> = a_array.dot(&b_array);

        let c_dense = c_sparse.to_dense().unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (c_dense.as_array()[[i, j]] - c_dense_expected[[i, j]]).abs() < 1e-10,
                    "Mismatch at [{}, {}]: sparse={}, dense={}",
                    i,
                    j,
                    c_dense.as_array()[[i, j]],
                    c_dense_expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_csc_spspmm_shape_mismatch() {
        let a = CscMatrix::<f64>::zeros((3, 4)).unwrap();
        let b = CscMatrix::<f64>::zeros((5, 2)).unwrap();

        let result = a.spspmm(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_spspmm_accumulation() {
        use scirs2_core::ndarray_ext::array;

        // Test that multiple products accumulate correctly
        // A = [[1, 1],    B = [[1, 1],
        //      [1, 1]]         [1, 1]]
        let a_array = array![[1.0, 1.0], [1.0, 1.0]];
        let b_array = array![[1.0, 1.0], [1.0, 1.0]];

        let a_dense = DenseND::from_array(a_array.into_dyn());
        let b_dense = DenseND::from_array(b_array.into_dyn());

        let a = CscMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CscMatrix::from_dense(&b_dense, 1e-10).unwrap();

        let c = a.spspmm(&b).unwrap();

        // All entries should be 2 (1*1 + 1*1)
        let c_dense = c.to_dense().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((c_dense.as_array()[[i, j]] - 2.0).abs() < 1e-10);
            }
        }
    }
}
