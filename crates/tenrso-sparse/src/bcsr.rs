//! BCSR (Block Compressed Sparse Row) format
//!
//! Block-based sparse matrix storage where non-zero blocks are stored as dense sub-matrices.
//! This format is efficient for matrices with block structure and provides better cache locality
//! than CSR for block-structured operations.
//!
//! # Structure
//!
//! - **block_shape**: (r, c) - dimensions of each dense block
//! - **row_ptr**: Pointers to start of each block row
//! - **col_indices**: Block column indices for each non-zero block
//! - **blocks**: Dense block data stored contiguously (row-major within each block)
//!
//! # Use Cases
//!
//! - Matrices with natural block structure (e.g., finite element methods)
//! - Better cache locality for block operations
//! - SIMD and tensor core optimizations
//! - GPU acceleration with block-structured data
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray_ext::Array2;
//! use tenrso_sparse::bcsr::BcsrMatrix;
//!
//! // Create a 4×4 matrix with 2×2 blocks
//! let dense = Array2::from_shape_vec((4, 4), vec![
//!     1.0, 2.0, 0.0, 0.0,
//!     3.0, 4.0, 0.0, 0.0,
//!     0.0, 0.0, 5.0, 6.0,
//!     0.0, 0.0, 7.0, 8.0,
//! ]).unwrap();
//!
//! let bcsr = BcsrMatrix::from_dense(&dense.view(), (2, 2), 1e-10).unwrap();
//!
//! assert_eq!(bcsr.nnz(), 8); // 2 blocks × 4 elements
//! assert_eq!(bcsr.num_blocks(), 2);
//! ```

use crate::csr::CsrMatrix;
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BcsrError {
    #[error("Invalid block shape: {0}")]
    InvalidBlockShape(String),

    #[error("Matrix shape {matrix_shape:?} not divisible by block shape {block_shape:?}")]
    ShapeNotDivisible {
        matrix_shape: (usize, usize),
        block_shape: (usize, usize),
    },

    #[error("Invalid row pointers: length {len} for {nrows} block rows (expected {expected})")]
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

    #[error("Length mismatch: {col_indices} col_indices but {blocks} blocks")]
    LengthMismatch { col_indices: usize, blocks: usize },

    #[error("Block column index out of bounds: {col_idx} >= {ncols}")]
    ColIndexOutOfBounds { col_idx: usize, ncols: usize },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

/// Block Compressed Sparse Row (BCSR) matrix
///
/// Stores sparse matrices as a collection of dense blocks.
/// More efficient than CSR for block-structured matrices.
#[derive(Debug, Clone)]
pub struct BcsrMatrix<T> {
    /// Block shape (rows, cols) - all blocks have the same dimensions
    block_shape: (usize, usize),

    /// Row pointers for block rows (length = num_block_rows + 1)
    row_ptr: Vec<usize>,

    /// Block column indices (length = num_blocks)
    col_indices: Vec<usize>,

    /// Dense block data stored contiguously (length = num_blocks * block_size)
    /// Blocks stored in row-major order within each block
    blocks: Vec<T>,

    /// Matrix shape in elements (not blocks)
    shape: (usize, usize),
}

impl<T: Float> BcsrMatrix<T> {
    /// Create a new BCSR matrix with validation
    ///
    /// # Arguments
    ///
    /// * `block_shape` - (rows, cols) dimensions of each block
    /// * `row_ptr` - Block row pointers
    /// * `col_indices` - Block column indices
    /// * `blocks` - Dense block data
    /// * `shape` - Matrix shape in elements
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Block shape is invalid (zero dimensions)
    /// - Matrix shape not divisible by block shape
    /// - Row pointers invalid or not sorted
    /// - Length mismatches between data structures
    /// - Column indices out of bounds
    pub fn new(
        block_shape: (usize, usize),
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        blocks: Vec<T>,
        shape: (usize, usize),
    ) -> Result<Self, BcsrError> {
        // Validate block shape
        if block_shape.0 == 0 || block_shape.1 == 0 {
            return Err(BcsrError::InvalidBlockShape(
                "Block dimensions cannot be zero".to_string(),
            ));
        }

        // Validate matrix shape is divisible by block shape
        if !shape.0.is_multiple_of(block_shape.0) || !shape.1.is_multiple_of(block_shape.1) {
            return Err(BcsrError::ShapeNotDivisible {
                matrix_shape: shape,
                block_shape,
            });
        }

        let num_block_rows = shape.0 / block_shape.0;
        let num_block_cols = shape.1 / block_shape.1;

        // Validate row_ptr length
        if row_ptr.len() != num_block_rows + 1 {
            return Err(BcsrError::InvalidRowPtr {
                len: row_ptr.len(),
                nrows: num_block_rows,
                expected: num_block_rows + 1,
            });
        }

        // Validate row_ptr is sorted
        for i in 0..row_ptr.len() - 1 {
            if row_ptr[i] > row_ptr[i + 1] {
                return Err(BcsrError::RowPtrNotSorted {
                    idx: i,
                    curr: row_ptr[i],
                    next: row_ptr[i + 1],
                });
            }
        }

        let num_blocks = col_indices.len();
        let block_size = block_shape.0 * block_shape.1;

        // Validate lengths match
        if blocks.len() != num_blocks * block_size {
            return Err(BcsrError::LengthMismatch {
                col_indices: num_blocks,
                blocks: blocks.len() / block_size,
            });
        }

        // Validate column indices
        for &col_idx in &col_indices {
            if col_idx >= num_block_cols {
                return Err(BcsrError::ColIndexOutOfBounds {
                    col_idx,
                    ncols: num_block_cols,
                });
            }
        }

        Ok(Self {
            block_shape,
            row_ptr,
            col_indices,
            blocks,
            shape,
        })
    }

    /// Create an empty BCSR matrix with given shape and block size
    pub fn zeros(shape: (usize, usize), block_shape: (usize, usize)) -> Result<Self, BcsrError> {
        if block_shape.0 == 0 || block_shape.1 == 0 {
            return Err(BcsrError::InvalidBlockShape(
                "Block dimensions cannot be zero".to_string(),
            ));
        }

        if !shape.0.is_multiple_of(block_shape.0) || !shape.1.is_multiple_of(block_shape.1) {
            return Err(BcsrError::ShapeNotDivisible {
                matrix_shape: shape,
                block_shape,
            });
        }

        let num_block_rows = shape.0 / block_shape.0;
        let row_ptr = vec![0; num_block_rows + 1];

        Ok(Self {
            block_shape,
            row_ptr,
            col_indices: Vec::new(),
            blocks: Vec::new(),
            shape,
        })
    }

    /// Number of non-zero elements (not blocks)
    pub fn nnz(&self) -> usize {
        self.blocks.len()
    }

    /// Number of non-zero blocks
    pub fn num_blocks(&self) -> usize {
        self.col_indices.len()
    }

    /// Matrix shape in elements
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

    /// Block shape
    pub fn block_shape(&self) -> (usize, usize) {
        self.block_shape
    }

    /// Number of block rows
    pub fn num_block_rows(&self) -> usize {
        self.shape.0 / self.block_shape.0
    }

    /// Number of block columns
    pub fn num_block_cols(&self) -> usize {
        self.shape.1 / self.block_shape.1
    }

    /// Row pointers
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Block column indices
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Block data
    pub fn blocks(&self) -> &[T] {
        &self.blocks
    }

    /// Density (nnz / total)
    pub fn density(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        self.nnz() as f64 / total as f64
    }

    /// Get a specific block as a view
    ///
    /// Returns None if block doesn't exist (is zero).
    pub fn get_block(&self, block_row: usize, block_col: usize) -> Option<Array2<T>> {
        let start = self.row_ptr[block_row];
        let end = self.row_ptr[block_row + 1];

        for idx in start..end {
            if self.col_indices[idx] == block_col {
                let block_size = self.block_shape.0 * self.block_shape.1;
                let block_start = idx * block_size;
                let block_data = &self.blocks[block_start..block_start + block_size];

                return Some(
                    Array2::from_shape_vec(self.block_shape, block_data.to_vec()).unwrap(),
                );
            }
        }

        None
    }

    /// Create BCSR matrix from dense matrix
    ///
    /// Divides dense matrix into blocks and stores only non-zero blocks.
    /// A block is considered non-zero if any element exceeds threshold.
    pub fn from_dense(
        dense: &ArrayView2<T>,
        block_shape: (usize, usize),
        threshold: T,
    ) -> Result<Self, BcsrError> {
        let (rows, cols) = (dense.nrows(), dense.ncols());

        if block_shape.0 == 0 || block_shape.1 == 0 {
            return Err(BcsrError::InvalidBlockShape(
                "Block dimensions cannot be zero".to_string(),
            ));
        }

        if rows % block_shape.0 != 0 || cols % block_shape.1 != 0 {
            return Err(BcsrError::ShapeNotDivisible {
                matrix_shape: (rows, cols),
                block_shape,
            });
        }

        let num_block_rows = rows / block_shape.0;
        let num_block_cols = cols / block_shape.1;
        let block_size = block_shape.0 * block_shape.1;

        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut blocks = Vec::new();

        for block_row in 0..num_block_rows {
            let row_start = block_row * block_shape.0;

            for block_col in 0..num_block_cols {
                let col_start = block_col * block_shape.1;

                // Extract block
                let mut block_data = Vec::with_capacity(block_size);
                let mut has_nonzero = false;

                for i in 0..block_shape.0 {
                    for j in 0..block_shape.1 {
                        let val = dense[[row_start + i, col_start + j]];
                        block_data.push(val);
                        if val.abs() > threshold {
                            has_nonzero = true;
                        }
                    }
                }

                // Store block if non-zero
                if has_nonzero {
                    col_indices.push(block_col);
                    blocks.extend(block_data);
                }
            }

            row_ptr.push(col_indices.len());
        }

        Self::new(block_shape, row_ptr, col_indices, blocks, (rows, cols))
    }

    /// Convert BCSR matrix to dense matrix
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = Array2::zeros(self.shape);
        let block_size = self.block_shape.0 * self.block_shape.1;

        for block_row in 0..self.num_block_rows() {
            let start = self.row_ptr[block_row];
            let end = self.row_ptr[block_row + 1];

            for idx in start..end {
                let block_col = self.col_indices[idx];
                let block_start = idx * block_size;
                let block_data = &self.blocks[block_start..block_start + block_size];

                let row_start = block_row * self.block_shape.0;
                let col_start = block_col * self.block_shape.1;

                // Copy block data to dense matrix
                let mut data_idx = 0;
                for i in 0..self.block_shape.0 {
                    for j in 0..self.block_shape.1 {
                        dense[[row_start + i, col_start + j]] = block_data[data_idx];
                        data_idx += 1;
                    }
                }
            }
        }

        dense
    }

    /// Convert BCSR to CSR format
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let mut csr_row_ptr = vec![0];
        let mut csr_col_indices = Vec::new();
        let mut csr_values = Vec::new();

        let block_size = self.block_shape.0 * self.block_shape.1;

        for block_row in 0..self.num_block_rows() {
            for local_row in 0..self.block_shape.0 {
                let start = self.row_ptr[block_row];
                let end = self.row_ptr[block_row + 1];

                for idx in start..end {
                    let block_col = self.col_indices[idx];
                    let block_start = idx * block_size;

                    for local_col in 0..self.block_shape.1 {
                        let elem_idx = local_row * self.block_shape.1 + local_col;
                        let val = self.blocks[block_start + elem_idx];

                        if val != T::zero() {
                            csr_col_indices.push(block_col * self.block_shape.1 + local_col);
                            csr_values.push(val);
                        }
                    }
                }

                csr_row_ptr.push(csr_values.len());
            }
        }

        CsrMatrix::new(csr_row_ptr, csr_col_indices, csr_values, self.shape).unwrap()
    }

    /// Sparse matrix-vector multiply: y = A * x
    ///
    /// # Complexity
    ///
    /// Time: O(num_blocks * block_size)
    /// Space: O(nrows)
    pub fn spmv(&self, x: &ArrayView1<T>) -> Result<Array1<T>, BcsrError> {
        if x.len() != self.ncols() {
            return Err(BcsrError::ShapeMismatch(format!(
                "Vector length {} doesn't match matrix columns {}",
                x.len(),
                self.ncols()
            )));
        }

        let mut y = Array1::zeros(self.nrows());
        let block_size = self.block_shape.0 * self.block_shape.1;

        for block_row in 0..self.num_block_rows() {
            let start = self.row_ptr[block_row];
            let end = self.row_ptr[block_row + 1];

            for idx in start..end {
                let block_col = self.col_indices[idx];
                let block_start = idx * block_size;

                // Multiply block with corresponding part of x
                for i in 0..self.block_shape.0 {
                    let mut sum = T::zero();
                    for j in 0..self.block_shape.1 {
                        let block_elem = self.blocks[block_start + i * self.block_shape.1 + j];
                        let x_elem = x[block_col * self.block_shape.1 + j];
                        sum = sum + block_elem * x_elem;
                    }
                    y[block_row * self.block_shape.0 + i] =
                        y[block_row * self.block_shape.0 + i] + sum;
                }
            }
        }

        Ok(y)
    }

    /// Block Sparse Matrix-Matrix Multiply: C = A * B
    ///
    /// Computes the product of a BCSR matrix A with a dense matrix B.
    /// Uses block-wise multiplication for efficiency.
    ///
    /// # Arguments
    ///
    /// * `b` - Dense matrix B (n×k) to multiply with
    ///
    /// # Returns
    ///
    /// Dense result matrix C (m×k)
    ///
    /// # Complexity
    ///
    /// Time: O(num_blocks × r × c × k) where (r,c) is block_shape, k is B.ncols()
    /// Space: O(m × k)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray_ext::{Array2, array};
    /// use tenrso_sparse::bcsr::BcsrMatrix;
    ///
    /// // Create a 4×4 BCSR matrix with 2×2 blocks
    /// let a_dense = array![
    ///     [1.0, 2.0, 0.0, 0.0],
    ///     [3.0, 4.0, 0.0, 0.0],
    ///     [0.0, 0.0, 5.0, 6.0],
    ///     [0.0, 0.0, 7.0, 8.0],
    /// ];
    /// let a = BcsrMatrix::from_dense(&a_dense.view(), (2, 2), 1e-10).unwrap();
    ///
    /// // Create a dense 4×2 matrix B
    /// let b = array![
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    /// ];
    ///
    /// // Compute C = A * B
    /// let c = a.spmm(&b.view()).unwrap();
    ///
    /// assert_eq!(c.shape(), &[4, 2]);
    /// ```
    pub fn spmm(&self, b: &ArrayView2<T>) -> Result<Array2<T>, BcsrError> {
        // Validate dimensions
        if b.nrows() != self.ncols() {
            return Err(BcsrError::ShapeMismatch(format!(
                "Matrix shape mismatch: A is {}×{}, B is {}×{}",
                self.nrows(),
                self.ncols(),
                b.nrows(),
                b.ncols()
            )));
        }

        let m = self.nrows();
        let k = b.ncols();
        let mut c = Array2::zeros((m, k));

        let block_size = self.block_shape.0 * self.block_shape.1;
        let (r, c_block) = self.block_shape;

        // For each block row in A
        for block_row in 0..self.num_block_rows() {
            let start = self.row_ptr[block_row];
            let end = self.row_ptr[block_row + 1];

            // For each non-zero block in this block row
            for idx in start..end {
                let block_col = self.col_indices[idx];
                let block_start = idx * block_size;

                // Get the dense block (r × c_block)
                // This block affects rows [block_row*r .. (block_row+1)*r] in C
                // And multiplies with rows [block_col*c_block .. (block_col+1)*c_block] in B

                // For each row i in the block (0..r)
                for i in 0..r {
                    let c_row = block_row * r + i;

                    // For each column j_out in B (0..k)
                    for j_out in 0..k {
                        let mut sum = T::zero();

                        // For each column j in the block (0..c_block)
                        for j in 0..c_block {
                            let block_elem = self.blocks[block_start + i * c_block + j];
                            let b_row = block_col * c_block + j;
                            let b_elem = b[[b_row, j_out]];
                            sum = sum + block_elem * b_elem;
                        }

                        // Accumulate into C
                        c[[c_row, j_out]] = c[[c_row, j_out]] + sum;
                    }
                }
            }
        }

        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_bcsr_creation() {
        let block_shape = (2, 2);
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let blocks = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = (4, 4);

        let bcsr = BcsrMatrix::new(block_shape, row_ptr, col_indices, blocks, shape).unwrap();

        assert_eq!(bcsr.shape(), (4, 4));
        assert_eq!(bcsr.block_shape(), (2, 2));
        assert_eq!(bcsr.num_blocks(), 2);
        assert_eq!(bcsr.nnz(), 8);
    }

    #[test]
    fn test_bcsr_zeros() {
        let bcsr = BcsrMatrix::<f64>::zeros((4, 4), (2, 2)).unwrap();

        assert_eq!(bcsr.shape(), (4, 4));
        assert_eq!(bcsr.num_blocks(), 0);
        assert_eq!(bcsr.nnz(), 0);
    }

    #[test]
    fn test_bcsr_from_dense() {
        let dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let bcsr = BcsrMatrix::from_dense(&dense.view(), (2, 2), 1e-10).unwrap();

        assert_eq!(bcsr.num_blocks(), 2);
        assert_eq!(bcsr.nnz(), 8);

        let block0 = bcsr.get_block(0, 0).unwrap();
        assert_eq!(block0[[0, 0]], 1.0);
        assert_eq!(block0[[0, 1]], 2.0);
        assert_eq!(block0[[1, 0]], 3.0);
        assert_eq!(block0[[1, 1]], 4.0);
    }

    #[test]
    fn test_bcsr_to_dense() {
        let dense_orig = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let bcsr = BcsrMatrix::from_dense(&dense_orig.view(), (2, 2), 1e-10).unwrap();
        let dense_recovered = bcsr.to_dense();

        for i in 0..4 {
            for j in 0..4 {
                assert!((dense_orig[[i, j]] - dense_recovered[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bcsr_spmv() {
        let dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let bcsr = BcsrMatrix::from_dense(&dense.view(), (2, 2), 1e-10).unwrap();
        let x = array![1.0, 2.0, 3.0, 4.0];

        let y = bcsr.spmv(&x.view()).unwrap();

        // Expected: y = A * x
        let y_expected = dense.dot(&x);

        for i in 0..4 {
            assert!((y[i] - y_expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bcsr_get_block() {
        let dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let bcsr = BcsrMatrix::from_dense(&dense.view(), (2, 2), 1e-10).unwrap();

        let block11 = bcsr.get_block(1, 1).unwrap();
        assert_eq!(block11[[0, 0]], 5.0);
        assert_eq!(block11[[1, 1]], 8.0);

        // Non-existent block should return None
        let block01 = bcsr.get_block(0, 1);
        assert!(block01.is_none());
    }

    #[test]
    fn test_bcsr_invalid_block_shape() {
        let result = BcsrMatrix::<f64>::zeros((4, 4), (0, 2));
        assert!(result.is_err());

        let result = BcsrMatrix::<f64>::zeros((4, 4), (2, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_bcsr_shape_not_divisible() {
        let result = BcsrMatrix::<f64>::zeros((5, 5), (2, 2));
        assert!(result.is_err());
    }

    #[test]
    fn test_bcsr_to_csr() {
        let dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let bcsr = BcsrMatrix::from_dense(&dense.view(), (2, 2), 1e-10).unwrap();
        let csr = bcsr.to_csr();

        let dense_from_csr = csr.to_dense().unwrap();

        for i in 0..4 {
            for j in 0..4 {
                assert!((dense[[i, j]] - dense_from_csr.as_array()[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bcsr_spmm_basic() {
        // Create a 4×4 BCSR matrix with 2×2 blocks
        let a_dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let a = BcsrMatrix::from_dense(&a_dense.view(), (2, 2), 1e-10).unwrap();

        // Create a 4×2 dense matrix B
        let b = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],];

        // Compute C = A * B
        let c = a.spmm(&b.view()).unwrap();

        // Compute expected result using dense multiplication
        let c_expected = a_dense.dot(&b);

        // Verify shape
        assert_eq!(c.shape(), &[4, 2]);

        // Verify values
        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (c[[i, j]] - c_expected[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): {} != {}",
                    i,
                    j,
                    c[[i, j]],
                    c_expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_bcsr_spmm_identity() {
        // Create a 4×4 identity matrix as BCSR with 2×2 blocks
        let identity = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let a = BcsrMatrix::from_dense(&identity.view(), (2, 2), 1e-10).unwrap();

        // Create a 4×3 dense matrix B
        let b = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        // I * B = B
        let c = a.spmm(&b.view()).unwrap();

        for i in 0..4 {
            for j in 0..3 {
                assert!((c[[i, j]] - b[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bcsr_spmm_zeros() {
        // Create a zero BCSR matrix
        let a = BcsrMatrix::<f64>::zeros((4, 4), (2, 2)).unwrap();

        // Create a 4×3 dense matrix B
        let b = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        // 0 * B = 0
        let c = a.spmm(&b.view()).unwrap();

        for i in 0..4 {
            for j in 0..3 {
                assert!(c[[i, j]].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bcsr_spmm_shape_mismatch() {
        // Create a 4×4 BCSR matrix
        let a_dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let a = BcsrMatrix::from_dense(&a_dense.view(), (2, 2), 1e-10).unwrap();

        // Create a 3×2 dense matrix B (wrong size)
        let b = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];

        // Should error due to shape mismatch
        let result = a.spmm(&b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_bcsr_spmm_single_column() {
        // Create a 4×4 BCSR matrix
        let a_dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let a = BcsrMatrix::from_dense(&a_dense.view(), (2, 2), 1e-10).unwrap();

        // Create a 4×1 dense matrix B (single column)
        let b = array![[1.0], [2.0], [3.0], [4.0]];

        // Compute C = A * B
        let c = a.spmm(&b.view()).unwrap();

        // Expected result (should match SpMV)
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y_expected = a.spmv(&x.view()).unwrap();

        // Verify shape
        assert_eq!(c.shape(), &[4, 1]);

        // Verify values match SpMV result
        for i in 0..4 {
            assert!(
                (c[[i, 0]] - y_expected[i]).abs() < 1e-10,
                "Mismatch at row {}: {} != {}",
                i,
                c[[i, 0]],
                y_expected[i]
            );
        }
    }

    #[test]
    fn test_bcsr_spmm_wide_result() {
        // Create a 4×4 BCSR matrix
        let a_dense = array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ];

        let a = BcsrMatrix::from_dense(&a_dense.view(), (2, 2), 1e-10).unwrap();

        // Create a 4×8 dense matrix B (wide result)
        let b = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64);

        // Compute C = A * B
        let c = a.spmm(&b.view()).unwrap();

        // Compute expected result
        let c_expected = a_dense.dot(&b);

        // Verify shape
        assert_eq!(c.shape(), &[4, 8]);

        // Verify values
        for i in 0..4 {
            for j in 0..8 {
                assert!(
                    (c[[i, j]] - c_expected[[i, j]]).abs() < 1e-9,
                    "Mismatch at ({}, {}): {} != {}",
                    i,
                    j,
                    c[[i, j]],
                    c_expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_bcsr_spmm_correctness_vs_dense() {
        // Create a more complex BCSR matrix
        let a_dense = array![
            [1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 10.0, 11.0, 12.0],
            [0.0, 0.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 0.0, 0.0, 19.0, 20.0],
            [21.0, 22.0, 0.0, 0.0, 23.0, 24.0],
        ];

        let a = BcsrMatrix::from_dense(&a_dense.view(), (2, 2), 1e-10).unwrap();

        // Create a 6×4 dense matrix B
        let b = Array2::from_shape_fn((6, 4), |(i, j)| (i + j) as f64);

        // Compute C = A * B
        let c = a.spmm(&b.view()).unwrap();

        // Compute expected result
        let c_expected = a_dense.dot(&b);

        // Verify all values
        for i in 0..6 {
            for j in 0..4 {
                assert!(
                    (c[[i, j]] - c_expected[[i, j]]).abs() < 1e-9,
                    "Mismatch at ({}, {}): {} != {}",
                    i,
                    j,
                    c[[i, j]],
                    c_expected[[i, j]]
                );
            }
        }
    }
}
