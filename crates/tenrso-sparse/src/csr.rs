//! CSR (Compressed Sparse Row) format for 2D matrices
//!
//! CSR is optimized for row-wise operations and is the standard format for sparse matrices.
//!
//! # Format
//!
//! For an m×n sparse matrix with nnz non-zeros:
//! - `row_ptr`: `Vec<usize>` of length m+1 - row_ptr\[i\] points to start of row i
//! - `col_indices`: `Vec<usize>` of length nnz - column index for each non-zero
//! - `values`: `Vec<T>` of length nnz - the non-zero values
//! - `shape`: (m, n) - dimensions of the matrix
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::csr::CsrMatrix;
//!
//! // Create a 3×4 sparse matrix:
//! // [1.0  0   2.0  0  ]
//! // [0    3.0 0    0  ]
//! // [4.0  0   0    5.0]
//!
//! let row_ptr = vec![0, 2, 3, 5];  // Cumulative row starts
//! let col_indices = vec![0, 2, 1, 0, 3];
//! let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 4);
//!
//! let csr = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();
//! assert_eq!(csr.nnz(), 5);
//! ```
//!
//! # SciRS2 Integration
//!
//! All operations use `scirs2_core` types. Direct use of `ndarray` is forbidden.

use crate::coo::{CooError, CooTensor};
use crate::csc::CscMatrix;
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CsrError {
    #[error("Invalid row pointers: length {len} for {nrows} rows (expected {expected})")]
    InvalidRowPtr {
        len: usize,
        nrows: usize,
        expected: usize,
    },

    #[error("Row pointer not sorted at index {idx}: {curr} > {next}")]
    RowPtrNotSorted {
        idx: usize,
        curr: usize,
        next: usize,
    },

    #[error("Length mismatch: {col_indices} col_indices but {values} values")]
    LengthMismatch { col_indices: usize, values: usize },

    #[error("Column index out of bounds: {col_idx} >= {ncols}")]
    ColIndexOutOfBounds { col_idx: usize, ncols: usize },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Shape mismatch: matrix is {nrows}×{ncols}, vector has length {vec_len}")]
    ShapeMismatch {
        nrows: usize,
        ncols: usize,
        vec_len: usize,
    },

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

/// CSR (Compressed Sparse Row) matrix
///
/// Optimized for row-wise operations like SpMV (Sparse Matrix-Vector multiply).
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    /// Row pointers: row_ptr[i] = start index of row i in col_indices/values
    /// Length: nrows + 1, with row_ptr[nrows] = nnz
    row_ptr: Vec<usize>,

    /// Column indices for each non-zero element
    col_indices: Vec<usize>,

    /// Values of non-zero elements
    values: Vec<T>,

    /// Shape: (nrows, ncols)
    shape: (usize, usize),
}

impl<T: Clone> CsrMatrix<T> {
    /// Create a new CSR matrix
    ///
    /// # Arguments
    ///
    /// * `row_ptr` - Row pointers (length nrows+1)
    /// * `col_indices` - Column indices for each non-zero
    /// * `values` - Values for each non-zero
    /// * `shape` - (nrows, ncols)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - row_ptr length is incorrect
    /// - col_indices and values have different lengths
    /// - row_ptr is not monotonically increasing
    /// - any column index is out of bounds
    pub fn new(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
        shape: (usize, usize),
    ) -> Result<Self, CsrError> {
        let (nrows, ncols) = shape;

        // Validate shape
        if nrows == 0 || ncols == 0 {
            return Err(CsrError::InvalidShape(
                "Shape cannot have zeros".to_string(),
            ));
        }

        // Validate row_ptr length
        if row_ptr.len() != nrows + 1 {
            return Err(CsrError::InvalidRowPtr {
                len: row_ptr.len(),
                nrows,
                expected: nrows + 1,
            });
        }

        // Validate col_indices and values length
        if col_indices.len() != values.len() {
            return Err(CsrError::LengthMismatch {
                col_indices: col_indices.len(),
                values: values.len(),
            });
        }

        // Validate row_ptr is monotonically increasing
        for i in 0..nrows {
            if row_ptr[i] > row_ptr[i + 1] {
                return Err(CsrError::RowPtrNotSorted {
                    idx: i,
                    curr: row_ptr[i],
                    next: row_ptr[i + 1],
                });
            }
        }

        // Validate final row_ptr matches nnz
        let nnz = col_indices.len();
        if row_ptr[nrows] != nnz {
            return Err(CsrError::InvalidRowPtr {
                len: row_ptr[nrows],
                nrows,
                expected: nnz,
            });
        }

        // Validate column indices
        for &col_idx in &col_indices {
            if col_idx >= ncols {
                return Err(CsrError::ColIndexOutOfBounds { col_idx, ncols });
            }
        }

        Ok(Self {
            row_ptr,
            col_indices,
            values,
            shape,
        })
    }

    /// Create an empty CSR matrix with given shape
    pub fn zeros(shape: (usize, usize)) -> Result<Self, CsrError> {
        let (nrows, ncols) = shape;
        if nrows == 0 || ncols == 0 {
            return Err(CsrError::InvalidShape(
                "Shape cannot have zeros".to_string(),
            ));
        }

        Ok(Self {
            row_ptr: vec![0; nrows + 1],
            col_indices: Vec::new(),
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

    /// Get row pointers
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Get column indices
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
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

    /// Get a row as (col_indices, values) slices
    pub fn row(&self, i: usize) -> Option<(&[usize], &[T])> {
        if i >= self.nrows() {
            return None;
        }

        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];

        Some((&self.col_indices[start..end], &self.values[start..end]))
    }
}

impl<T: Float> CsrMatrix<T> {
    /// Convert from COO format
    ///
    /// The input COO tensor must be 2-dimensional.
    pub fn from_coo(coo: &CooTensor<T>) -> Result<Self, CsrError> {
        if coo.rank() != 2 {
            return Err(CsrError::InvalidShape(format!(
                "COO tensor must be 2D, got {}D",
                coo.rank()
            )));
        }

        let nrows = coo.shape()[0];
        let ncols = coo.shape()[1];
        let nnz = coo.nnz();

        // Sort COO by row then column
        let mut coo_sorted = coo.clone();
        coo_sorted.sort();

        // Build CSR
        let mut row_ptr = vec![0; nrows + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Count elements per row
        for idx in coo_sorted.indices() {
            let row = idx[0];
            row_ptr[row + 1] += 1;
        }

        // Cumulative sum to get row starts
        for i in 0..nrows {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Fill col_indices and values
        for (idx, &value) in coo_sorted.indices().iter().zip(coo_sorted.values()) {
            col_indices.push(idx[1]);
            values.push(value);
        }

        Self::new(row_ptr, col_indices, values, (nrows, ncols))
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> CooTensor<T> {
        let mut indices = Vec::with_capacity(self.nnz());
        let values = self.values.clone();

        for row in 0..self.nrows() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for col_idx in start..end {
                let col = self.col_indices[col_idx];
                indices.push(vec![row, col]);
            }
        }

        let shape = vec![self.nrows(), self.ncols()];
        CooTensor::new(indices, values, shape).unwrap()
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Result<DenseND<T>> {
        let (nrows, ncols) = self.shape;
        let mut data = vec![T::zero(); nrows * ncols];

        for row in 0..nrows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let value = self.values[idx];
                data[row * ncols + col] = value;
            }
        }

        DenseND::from_vec(data, &[nrows, ncols])
    }

    /// Create CSR from dense matrix
    ///
    /// Only stores elements where |value| > threshold.
    pub fn from_dense(dense: &DenseND<T>, threshold: T) -> Result<Self, CsrError> {
        if dense.rank() != 2 {
            return Err(CsrError::InvalidShape(format!(
                "Dense tensor must be 2D, got {}D",
                dense.rank()
            )));
        }

        // First convert to COO, then to CSR
        let coo = CooTensor::from_dense(dense, threshold);
        Self::from_coo(&coo)
    }

    /// Convert to CSC format (transpose)
    ///
    /// Converts CSR(A) to CSC(A^T). This is efficient because CSR's rows
    /// become CSC's columns directly.
    ///
    /// # Returns
    ///
    /// CSC matrix representing the transpose of this matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::csr::CsrMatrix;
    /// use tenrso_core::DenseND;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Create CSR matrix
    /// let dense = DenseND::from_array(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    /// let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();
    ///
    /// // Convert to CSC (transpose)
    /// let csc = csr.to_csc();
    ///
    /// // CSC represents the transpose
    /// assert_eq!(csc.shape(), (2, 2));
    /// ```
    pub fn to_csc(&self) -> CscMatrix<T> {
        let (m, n) = self.shape;
        let nnz = self.nnz();

        // CSR(A) -> CSC(A^T) where A is (m, n)
        // Result CSC has shape (n, m)
        // CSR entry (row=i, col=j, val=v) becomes CSC entry at column i, row j

        // Count entries per column in result CSC
        let mut col_counts = vec![0; m]; // m columns in CSC (one per CSR row)
        for (row, count) in col_counts.iter_mut().enumerate().take(m) {
            let row_nnz = self.row_ptr[row + 1] - self.row_ptr[row];
            *count = row_nnz;
        }

        // Build col_ptr
        let mut col_ptr = vec![0; m + 1];
        for i in 0..m {
            col_ptr[i + 1] = col_ptr[i] + col_counts[i];
        }

        // Build row_indices and values
        let mut row_indices = vec![0; nnz];
        let mut values = vec![T::zero(); nnz];

        // Track current position in each CSC column
        let mut col_positions = col_ptr[0..m].to_vec();

        // Iterate through CSR and place entries into CSC
        for csr_row in 0..m {
            for idx in self.row_ptr[csr_row]..self.row_ptr[csr_row + 1] {
                let csr_col = self.col_indices[idx];
                let val = self.values[idx];

                // CSR entry (csr_row, csr_col) -> CSC entry in column csr_row, row csr_col
                let csc_col = csr_row;
                let csc_row = csr_col;

                let pos = col_positions[csc_col];
                row_indices[pos] = csc_row;
                values[pos] = val;
                col_positions[csc_col] += 1;
            }
        }

        CscMatrix::new(col_ptr, row_indices, values, (n, m)).unwrap()
    }

    /// Create CSR from CSC format (transpose)
    ///
    /// Converts CSC(A) to CSR(A^T). This is efficient because CSC's columns
    /// become CSR's rows directly.
    ///
    /// # Arguments
    ///
    /// * `csc` - CSC matrix to convert from
    ///
    /// # Returns
    ///
    /// CSR matrix representing the transpose of the input CSC matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::csc::CscMatrix;
    /// use tenrso_sparse::csr::CsrMatrix;
    /// use tenrso_core::DenseND;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Create CSC matrix
    /// let dense = DenseND::from_array(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    /// let csc = CscMatrix::from_dense(&dense, 1e-10).unwrap();
    ///
    /// // Convert to CSR (transpose)
    /// let csr = CsrMatrix::from_csc(&csc);
    ///
    /// // CSR represents the transpose
    /// assert_eq!(csr.shape(), (2, 2));
    /// ```
    pub fn from_csc(csc: &CscMatrix<T>) -> Self {
        let (csc_nrows, csc_ncols) = csc.shape();
        let nnz = csc.nnz();

        // CSC(A) -> CSR(A^T) where A is (m, n)
        // CSC has shape (m, n), result CSR has shape (n, m)
        // CSC entry (row=i, col=j) becomes CSR entry (row=j, col=i)

        let (m, n) = (csc_ncols, csc_nrows); // Transposed shape

        // Count entries per row in result CSR
        let mut row_counts = vec![0; m]; // m rows in CSR (one per CSC column)
        for (col, count) in row_counts.iter_mut().enumerate().take(csc_ncols) {
            let col_nnz = csc.col_ptr()[col + 1] - csc.col_ptr()[col];
            *count = col_nnz;
        }

        // Build row_ptr
        let mut row_ptr = vec![0; m + 1];
        for i in 0..m {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        // Build col_indices and values
        let mut col_indices = vec![0; nnz];
        let mut values = vec![T::zero(); nnz];

        // Track current position in each CSR row
        let mut row_positions = row_ptr[0..m].to_vec();

        // Iterate through CSC and place entries into CSR
        for csc_col in 0..csc_ncols {
            for idx in csc.col_ptr()[csc_col]..csc.col_ptr()[csc_col + 1] {
                let csc_row = csc.row_indices()[idx];
                let val = csc.values()[idx];

                // CSC entry (csc_row, csc_col) -> CSR entry (row=csc_col, col=csc_row)
                let csr_row = csc_col;
                let csr_col = csc_row;

                let pos = row_positions[csr_row];
                col_indices[pos] = csr_col;
                values[pos] = val;
                row_positions[csr_row] += 1;
            }
        }

        Self::new(row_ptr, col_indices, values, (m, n)).unwrap()
    }

    /// Sparse Matrix-Vector product: y = A * x
    ///
    /// Computes the product of this sparse matrix with a dense vector.
    ///
    /// # Arguments
    ///
    /// * `x` - Dense vector of length ncols
    ///
    /// # Returns
    ///
    /// Dense vector y of length nrows where y\[i\] = sum(A\[i,j\] * x\[j\])
    ///
    /// # Errors
    ///
    /// Returns error if x.len() != ncols
    ///
    /// # Complexity
    ///
    /// O(nnz) - linear in number of non-zeros
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::csr::CsrMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Matrix: [1 0 2]
    /// //         [0 3 0]
    /// let row_ptr = vec![0, 2, 3];
    /// let col_indices = vec![0, 2, 1];
    /// let values = vec![1.0, 2.0, 3.0];
    /// let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();
    ///
    /// let x = array![1.0, 2.0, 3.0];
    /// let y = csr.spmv(&x.view()).unwrap();
    /// assert_eq!(y[0], 7.0);  // 1*1 + 2*3
    /// assert_eq!(y[1], 6.0);  // 3*2
    /// ```
    pub fn spmv(&self, x: &ArrayView1<T>) -> Result<Array1<T>, CsrError> {
        // Validate dimensions
        if x.len() != self.ncols() {
            return Err(CsrError::ShapeMismatch {
                nrows: self.nrows(),
                ncols: self.ncols(),
                vec_len: x.len(),
            });
        }

        // Allocate result vector
        let mut y = Array1::<T>::zeros(self.nrows());

        // Compute y = A * x
        for row in 0..self.nrows() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            let mut sum = T::zero();
            for idx in start..end {
                let col = self.col_indices[idx];
                let value = self.values[idx];
                sum = sum + value * x[col];
            }
            y[row] = sum;
        }

        Ok(y)
    }

    /// Sparse Matrix-Matrix product: C = A * B
    ///
    /// Computes the product of this sparse matrix (A) with a dense matrix (B).
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
    /// use tenrso_sparse::csr::CsrMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Sparse matrix A: [1 0 2]
    /// //                  [0 3 0]
    /// let row_ptr = vec![0, 2, 3];
    /// let col_indices = vec![0, 2, 1];
    /// let values = vec![1.0, 2.0, 3.0];
    /// let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();
    ///
    /// // Dense matrix B: [1 2]
    /// //                 [3 4]
    /// //                 [5 6]
    /// let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    ///
    /// let c = csr.spmm(&b.view()).unwrap();
    /// assert_eq!(c[[0, 0]], 11.0);  // 1*1 + 2*5
    /// assert_eq!(c[[0, 1]], 14.0);  // 1*2 + 2*6
    /// assert_eq!(c[[1, 0]], 9.0);   // 3*3
    /// assert_eq!(c[[1, 1]], 12.0);  // 3*4
    /// ```
    pub fn spmm(&self, b: &ArrayView2<T>) -> Result<Array2<T>, CsrError> {
        let (b_rows, b_cols) = (b.nrows(), b.ncols());

        // Validate dimensions: A is (m, n), B is (n, k) -> C is (m, k)
        if self.ncols() != b_rows {
            return Err(CsrError::MatrixShapeMismatch {
                m1: self.nrows(),
                n1: self.ncols(),
                m2: b_rows,
                n2: b_cols,
            });
        }

        // Allocate result matrix
        let mut c = Array2::<T>::zeros((self.nrows(), b_cols));

        // Compute C = A * B by treating B as collection of column vectors
        // For each column k in B, compute C[:, k] = A * B[:, k]
        for k in 0..b_cols {
            let b_col = b.column(k);

            // Compute C[:, k] = A * B[:, k] using SpMV-like logic
            for row in 0..self.nrows() {
                let start = self.row_ptr[row];
                let end = self.row_ptr[row + 1];

                let mut sum = T::zero();
                for idx in start..end {
                    let col = self.col_indices[idx];
                    let value = self.values[idx];
                    sum = sum + value * b_col[col];
                }
                c[[row, k]] = sum;
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
    /// * `b` - Right-hand sparse matrix
    ///
    /// # Returns
    ///
    /// A new CSR matrix representing the product
    ///
    /// # Complexity
    ///
    /// Time: O(m × nnz_per_row_A × nnz_per_row_B)
    /// Space: O(nnz_result)
    ///
    /// Uses hash-based accumulation for efficient sparse result construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::csr::CsrMatrix;
    /// use tenrso_core::DenseND;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Create two sparse matrices
    /// let a_dense = DenseND::from_array(array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]].into_dyn());
    /// let b_dense = DenseND::from_array(array![[1.0, 0.0], [0.0, 0.0], [0.0, 4.0]].into_dyn());
    ///
    /// let a = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();
    /// let b = CsrMatrix::from_dense(&b_dense, 1e-10).unwrap();
    ///
    /// let c = a.spspmm(&b).unwrap();
    ///
    /// assert_eq!(c.shape(), (2, 2));
    /// // Result: [[1.0, 8.0], [0.0, 0.0]]
    /// ```
    pub fn spspmm(&self, b: &CsrMatrix<T>) -> Result<CsrMatrix<T>, CsrError> {
        // Validate dimensions: A is (m, n), B is (n, k) -> C is (m, k)
        if self.ncols() != b.nrows() {
            return Err(CsrError::MatrixShapeMismatch {
                m1: self.nrows(),
                n1: self.ncols(),
                m2: b.nrows(),
                n2: b.ncols(),
            });
        }

        let m = self.nrows();
        let k = b.ncols();

        // Use HashMap for each row to accumulate results
        use std::collections::HashMap;

        let mut result_row_ptr = vec![0];
        let mut result_col_indices = Vec::new();
        let mut result_values = Vec::new();

        // For each row in A
        for i in 0..m {
            let mut row_map: HashMap<usize, T> = HashMap::new();

            let a_row_start = self.row_ptr[i];
            let a_row_end = self.row_ptr[i + 1];

            // For each non-zero A[i, j]
            for a_idx in a_row_start..a_row_end {
                let j = self.col_indices[a_idx];
                let a_val = self.values[a_idx];

                // Get row j from B
                let b_row_start = b.row_ptr[j];
                let b_row_end = b.row_ptr[j + 1];

                // For each non-zero B[j, col]
                for b_idx in b_row_start..b_row_end {
                    let col = b.col_indices[b_idx];
                    let b_val = b.values[b_idx];

                    // Accumulate `A[i,j] * B[j,col]` into `result[i,col]`
                    let entry = row_map.entry(col).or_insert(T::zero());
                    *entry = *entry + a_val * b_val;
                }
            }

            // Convert row_map to sorted vectors
            let mut row_entries: Vec<_> = row_map.into_iter().collect();
            row_entries.sort_by_key(|(col, _)| *col);

            // Add non-zero entries to result
            for (col, val) in row_entries {
                if val != T::zero() {
                    result_col_indices.push(col);
                    result_values.push(val);
                }
            }

            result_row_ptr.push(result_col_indices.len());
        }

        CsrMatrix::new(result_row_ptr, result_col_indices, result_values, (m, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_creation() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 4);

        let csr = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), (3, 4));
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 4);
    }

    #[test]
    fn test_csr_zeros() {
        let csr = CsrMatrix::<f64>::zeros((5, 5)).unwrap();
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.shape(), (5, 5));
    }

    #[test]
    fn test_csr_row_access() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 4);

        let csr = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();

        let (cols, vals) = csr.row(0).unwrap();
        assert_eq!(cols, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);

        let (cols, vals) = csr.row(1).unwrap();
        assert_eq!(cols, &[1]);
        assert_eq!(vals, &[3.0]);

        let (cols, vals) = csr.row(2).unwrap();
        assert_eq!(cols, &[0, 3]);
        assert_eq!(vals, &[4.0, 5.0]);
    }

    #[test]
    fn test_csr_from_coo() {
        let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![2, 3]];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 4];

        let coo = CooTensor::new(indices, values, shape).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), (3, 4));
    }

    #[test]
    fn test_csr_to_coo() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 4);

        let csr = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();
        let coo = csr.to_coo();

        assert_eq!(coo.nnz(), 5);
        assert_eq!(coo.shape(), &[3, 4]);
    }

    #[test]
    fn test_csr_to_dense() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 4);

        let csr = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();
        let dense = csr.to_dense().unwrap();

        assert_eq!(dense.shape(), &[3, 4]);
        let view = dense.view();
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 2]], 2.0);
        assert_eq!(view[[1, 1]], 3.0);
        assert_eq!(view[[2, 0]], 4.0);
        assert_eq!(view[[2, 3]], 5.0);
    }

    #[test]
    fn test_csr_density() {
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let shape = (2, 10);

        let csr = CsrMatrix::new(row_ptr, col_indices, values, shape).unwrap();
        assert_eq!(csr.density(), 0.1); // 2/20
    }

    #[test]
    fn test_spmv_basic() {
        use scirs2_core::ndarray_ext::array;

        // Matrix: [1 0 2]
        //         [0 3 0]
        //         [4 0 5]
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = csr.spmv(&x.view()).unwrap();

        assert_eq!(y[0], 7.0); // 1*1 + 2*3 = 7
        assert_eq!(y[1], 6.0); // 3*2 = 6
        assert_eq!(y[2], 19.0); // 4*1 + 5*3 = 19
    }

    #[test]
    fn test_spmv_empty_rows() {
        use scirs2_core::ndarray_ext::array;

        // Matrix: [1 2 0]
        //         [0 0 0]  <- empty row
        //         [3 0 4]
        let row_ptr = vec![0, 2, 2, 4];
        let col_indices = vec![0, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = csr.spmv(&x.view()).unwrap();

        assert_eq!(y[0], 5.0); // 1*1 + 2*2 = 5
        assert_eq!(y[1], 0.0); // empty row
        assert_eq!(y[2], 15.0); // 3*1 + 4*3 = 15
    }

    #[test]
    fn test_spmv_shape_mismatch() {
        use scirs2_core::ndarray_ext::array;

        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        // Wrong size vector
        let x = array![1.0, 2.0]; // Should be length 3
        let result = csr.spmv(&x.view());

        assert!(result.is_err());
    }

    #[test]
    fn test_spmv_identity() {
        use scirs2_core::ndarray_ext::array;

        // Identity matrix
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 1.0, 1.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let x = array![5.0, 7.0, 9.0];
        let y = csr.spmv(&x.view()).unwrap();

        // Should return x itself
        assert_eq!(y[0], 5.0);
        assert_eq!(y[1], 7.0);
        assert_eq!(y[2], 9.0);
    }

    #[test]
    fn test_spmm_basic() {
        use scirs2_core::ndarray_ext::array;

        // Sparse matrix A: [1 0 2]
        //                  [0 3 0]
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        // Dense matrix B: [1 2]
        //                 [3 4]
        //                 [5 6]
        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let c = csr.spmm(&b.view()).unwrap();

        // Verify dimensions
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);

        // Verify values: C = A * B
        assert_eq!(c[[0, 0]], 11.0); // 1*1 + 2*5 = 11
        assert_eq!(c[[0, 1]], 14.0); // 1*2 + 2*6 = 14
        assert_eq!(c[[1, 0]], 9.0); // 3*3 = 9
        assert_eq!(c[[1, 1]], 12.0); // 3*4 = 12
    }

    #[test]
    fn test_spmm_single_column() {
        use scirs2_core::ndarray_ext::array;

        // Test SpMM with single column (should match SpMV)
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = array![[1.0], [2.0], [3.0]];
        let c = csr.spmm(&b.view()).unwrap();

        // Compare with SpMV result
        let x = array![1.0, 2.0, 3.0];
        let y = csr.spmv(&x.view()).unwrap();

        assert_eq!(c[[0, 0]], y[0]);
        assert_eq!(c[[1, 0]], y[1]);
        assert_eq!(c[[2, 0]], y[2]);
    }

    #[test]
    fn test_spmm_identity() {
        use scirs2_core::ndarray_ext::array;

        // Identity matrix
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 1.0, 1.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = csr.spmm(&b.view()).unwrap();

        // Should return B itself
        assert_eq!(c[[0, 0]], 1.0);
        assert_eq!(c[[0, 1]], 2.0);
        assert_eq!(c[[1, 0]], 3.0);
        assert_eq!(c[[1, 1]], 4.0);
        assert_eq!(c[[2, 0]], 5.0);
        assert_eq!(c[[2, 1]], 6.0);
    }

    #[test]
    fn test_spmm_empty_rows() {
        use scirs2_core::ndarray_ext::array;

        // Matrix with empty row
        let row_ptr = vec![0, 2, 2, 4];
        let col_indices = vec![0, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let c = csr.spmm(&b.view()).unwrap();

        // First row: 1*1 + 2*3 = 7, 1*2 + 2*4 = 10
        assert_eq!(c[[0, 0]], 7.0);
        assert_eq!(c[[0, 1]], 10.0);

        // Second row: empty, should be zero
        assert_eq!(c[[1, 0]], 0.0);
        assert_eq!(c[[1, 1]], 0.0);

        // Third row: 3*1 + 4*5 = 23, 3*2 + 4*6 = 30
        assert_eq!(c[[2, 0]], 23.0);
        assert_eq!(c[[2, 1]], 30.0);
    }

    #[test]
    fn test_spmm_shape_mismatch() {
        use scirs2_core::ndarray_ext::array;

        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        // Wrong size matrix: should be 3×k but is 2×2
        let b = array![[1.0, 2.0], [3.0, 4.0]];
        let result = csr.spmm(&b.view());

        assert!(result.is_err());
    }

    #[test]
    fn test_spmm_wide_result() {
        use scirs2_core::ndarray_ext::array;

        // Test with many columns in result
        let row_ptr = vec![0, 2];
        let col_indices = vec![0, 1];
        let values = vec![2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (1, 2)).unwrap();

        // B is 2×4
        let b = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let c = csr.spmm(&b.view()).unwrap();

        // Result should be 1×4
        assert_eq!(c.nrows(), 1);
        assert_eq!(c.ncols(), 4);

        // C = [2, 3] * [[1,2,3,4], [5,6,7,8]] = [2*1+3*5, 2*2+3*6, 2*3+3*7, 2*4+3*8]
        assert_eq!(c[[0, 0]], 17.0); // 2*1 + 3*5
        assert_eq!(c[[0, 1]], 22.0); // 2*2 + 3*6
        assert_eq!(c[[0, 2]], 27.0); // 2*3 + 3*7
        assert_eq!(c[[0, 3]], 32.0); // 2*4 + 3*8
    }

    #[test]
    fn test_spspmm_basic() {
        use scirs2_core::ndarray_ext::array;
        use tenrso_core::DenseND;

        // A = [[1, 0, 2],    B = [[1, 0],
        //      [0, 3, 0]]         [0, 0],
        //                         [0, 4]]
        let a_array = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let b_array = array![[1.0, 0.0], [0.0, 0.0], [0.0, 4.0]];

        let a_dense = DenseND::from_array(a_array.into_dyn());
        let b_dense = DenseND::from_array(b_array.into_dyn());

        let a = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CsrMatrix::from_dense(&b_dense, 1e-10).unwrap();

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
    fn test_spspmm_identity() {
        use scirs2_core::ndarray_ext::array;
        use tenrso_core::DenseND;

        // A = identity, B = sparse matrix
        let a_array = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b_array = array![[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]];

        let a_dense = DenseND::from_array(a_array.into_dyn());
        let b_dense = DenseND::from_array(b_array.into_dyn());

        let a = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CsrMatrix::from_dense(&b_dense, 1e-10).unwrap();

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
    fn test_spspmm_zeros() {
        // Test with all-zero matrices
        let a = CsrMatrix::<f64>::zeros((3, 3)).unwrap();
        let b = CsrMatrix::<f64>::zeros((3, 2)).unwrap();

        let c = a.spspmm(&b).unwrap();

        assert_eq!(c.nnz(), 0);
        assert_eq!(c.shape(), (3, 2));
    }

    #[test]
    fn test_spspmm_correctness_vs_dense() {
        use scirs2_core::ndarray_ext::array;
        use tenrso_core::DenseND;

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

        let a = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CsrMatrix::from_dense(&b_dense, 1e-10).unwrap();

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
    fn test_spspmm_shape_mismatch() {
        let a = CsrMatrix::<f64>::zeros((3, 4)).unwrap();
        let b = CsrMatrix::<f64>::zeros((5, 2)).unwrap();

        let result = a.spspmm(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_spspmm_accumulation() {
        use scirs2_core::ndarray_ext::array;
        use tenrso_core::DenseND;

        // Test that multiple products accumulate correctly
        // A = [[1, 1],    B = [[1, 1],
        //      [1, 1]]         [1, 1]]
        let a_array = array![[1.0, 1.0], [1.0, 1.0]];
        let b_array = array![[1.0, 1.0], [1.0, 1.0]];

        let a_dense = DenseND::from_array(a_array.into_dyn());
        let b_dense = DenseND::from_array(b_array.into_dyn());

        let a = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();
        let b = CsrMatrix::from_dense(&b_dense, 1e-10).unwrap();

        let c = a.spspmm(&b).unwrap();

        // All entries should be 2 (1*1 + 1*1)
        let c_dense = c.to_dense().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((c_dense.as_array()[[i, j]] - 2.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_csr_to_csc_basic() {
        use scirs2_core::ndarray_ext::array;

        // Create CSR matrix: [[1, 0, 2],
        //                     [0, 3, 0]]
        let a_array = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let a_dense = DenseND::from_array(a_array.into_dyn());
        let csr = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();

        // Convert to CSC (transpose)
        let csc = csr.to_csc();

        // CSC should be transpose: [[1, 0],
        //                           [0, 3],
        //                           [2, 0]]
        assert_eq!(csc.shape(), (3, 2)); // Transposed shape
        assert_eq!(csc.nnz(), 3);

        // Verify via dense conversion
        let csc_dense = csc.to_dense().unwrap();
        assert!((csc_dense.as_array()[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((csc_dense.as_array()[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((csc_dense.as_array()[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((csc_dense.as_array()[[1, 1]] - 3.0).abs() < 1e-10);
        assert!((csc_dense.as_array()[[2, 0]] - 2.0).abs() < 1e-10);
        assert!((csc_dense.as_array()[[2, 1]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_from_csc_basic() {
        use scirs2_core::ndarray_ext::array;

        // Create CSC matrix: [[1, 0],
        //                     [0, 2],
        //                     [3, 0]]
        let a_array = array![[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]];
        let a_dense = DenseND::from_array(a_array.into_dyn());
        let csc = CscMatrix::from_dense(&a_dense, 1e-10).unwrap();

        // Convert to CSR (transpose)
        let csr = CsrMatrix::from_csc(&csc);

        // CSR should be transpose: [[1, 0, 3],
        //                           [0, 2, 0]]
        assert_eq!(csr.shape(), (2, 3)); // Transposed shape
        assert_eq!(csr.nnz(), 3);

        // Verify via dense conversion
        let csr_dense = csr.to_dense().unwrap();
        assert!((csr_dense.as_array()[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((csr_dense.as_array()[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((csr_dense.as_array()[[0, 2]] - 3.0).abs() < 1e-10);
        assert!((csr_dense.as_array()[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((csr_dense.as_array()[[1, 1]] - 2.0).abs() < 1e-10);
        assert!((csr_dense.as_array()[[1, 2]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_csc_roundtrip() {
        use scirs2_core::ndarray_ext::array;

        // Create original CSR matrix
        let a_array = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 6.0, 0.0]
        ];
        let a_dense = DenseND::from_array(a_array.into_dyn());
        let csr_orig = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();

        // Roundtrip: CSR -> CSC -> CSR
        // Note: This gives us the double transpose, which should equal the original
        let csc = csr_orig.to_csc();
        let csr_back = CsrMatrix::from_csc(&csc);

        // The result should be the double transpose, which equals the original
        assert_eq!(csr_back.shape(), csr_orig.shape());
        assert_eq!(csr_back.nnz(), csr_orig.nnz());

        // Verify values match
        let orig_dense = csr_orig.to_dense().unwrap();
        let back_dense = csr_back.to_dense().unwrap();

        for i in 0..3 {
            for j in 0..4 {
                assert!(
                    (orig_dense.as_array()[[i, j]] - back_dense.as_array()[[i, j]]).abs() < 1e-10,
                    "Mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_csr_to_csc_transpose_correctness() {
        use scirs2_core::ndarray_ext::array;

        // Test that CSR -> CSC actually transposes
        let a_array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let a_dense = DenseND::from_array(a_array.clone().into_dyn());
        let csr = CsrMatrix::from_dense(&a_dense, 1e-10).unwrap();

        // Convert to CSC
        let csc = csr.to_csc();

        // CSC to dense should give transpose
        let csc_dense = csc.to_dense().unwrap();

        // Manually compute transpose and compare
        let a_transpose = a_array.t().to_owned();
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (csc_dense.as_array()[[i, j]] - a_transpose[[i, j]]).abs() < 1e-10,
                    "Transpose mismatch at [{}, {}]: got {}, expected {}",
                    i,
                    j,
                    csc_dense.as_array()[[i, j]],
                    a_transpose[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_csr_csc_empty_matrix() {
        // Test with zero matrix
        let csr = CsrMatrix::<f64>::zeros((3, 4)).unwrap();
        let csc = csr.to_csc();

        assert_eq!(csc.shape(), (4, 3)); // Transposed
        assert_eq!(csc.nnz(), 0);

        // Roundtrip
        let csr_back = CsrMatrix::from_csc(&csc);
        assert_eq!(csr_back.shape(), (3, 4));
        assert_eq!(csr_back.nnz(), 0);
    }

    #[test]
    fn test_csr_csc_identity() {
        use scirs2_core::ndarray_ext::array;

        // Test with identity matrix
        let id_array = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let id_dense = DenseND::from_array(id_array.into_dyn());
        let csr = CsrMatrix::from_dense(&id_dense, 1e-10).unwrap();

        // Identity transpose is identity
        let csc = csr.to_csc();
        let csr_back = CsrMatrix::from_csc(&csc);

        let orig_dense = csr.to_dense().unwrap();
        let back_dense = csr_back.to_dense().unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (orig_dense.as_array()[[i, j]] - back_dense.as_array()[[i, j]]).abs() < 1e-10
                );
            }
        }
    }
}
