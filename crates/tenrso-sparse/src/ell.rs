//! ELL (ELLPACK) Sparse Matrix Format
//!
//! The ELLPACK format stores sparse matrices with a fixed number of elements per row.
//! This format is particularly efficient for GPU operations and when rows have
//! similar numbers of non-zeros.
//!
//! # Structure
//!
//! - `data`: 2D array of values `(nrows × max_nnz_per_row)`
//! - `indices`: 2D array of column indices `(nrows × max_nnz_per_row)`
//! - Padded entries use sentinel value `ncols` for indices and `0.0` for values
//!
//! # Use Cases
//!
//! - GPU computations (coalesced memory access)
//! - Matrices with uniform row sparsity
//! - Vectorized operations (SIMD-friendly)
//!
//! # Performance
//!
//! - **SpMV**: O(nrows × max_nnz_per_row) - very efficient on GPUs
//! - **Memory**: Can waste space if rows have varying nnz
//! - **Best for**: Matrices where most rows have similar nnz
//!
//! # Example
//!
//! ```
//! use tenrso_sparse::ell::EllMatrix;
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create a small sparse matrix
//! let data = array![[1.0, 2.0], [3.0, 0.0], [4.0, 5.0]];
//! let indices = array![[0, 1], [1, 3], [0, 2]]; // 3 = sentinel (ncols)
//!
//! let ell = EllMatrix::new(data, indices, 3, 3).unwrap();
//! assert_eq!(ell.nrows(), 3);
//! assert_eq!(ell.ncols(), 3);
//! assert_eq!(ell.max_nnz_per_row(), 2);
//! ```

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt;

/// ELL (ELLPACK) sparse matrix format.
///
/// Stores sparse matrices with a fixed number of elements per row.
/// Efficient for GPU operations and uniform row sparsity.
#[derive(Debug, Clone)]
pub struct EllMatrix<T> {
    /// Value data `(nrows × max_nnz_per_row)`
    data: Array2<T>,
    /// Column indices `(nrows × max_nnz_per_row)`
    /// Padded entries use `ncols` as sentinel
    indices: Array2<usize>,
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
    /// Maximum non-zeros per row
    max_nnz_per_row: usize,
}

impl<T: Float> EllMatrix<T> {
    /// Creates a new ELL matrix from data and indices arrays.
    ///
    /// # Arguments
    ///
    /// - `data`: Value array `(nrows × max_nnz_per_row)`
    /// - `indices`: Column index array `(nrows × max_nnz_per_row)`
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are inconsistent or indices are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::ell::EllMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let data = array![[1.0, 2.0], [3.0, 0.0]];
    /// let indices = array![[0, 1], [1, 2]]; // 2 = sentinel
    /// let ell = EllMatrix::new(data, indices, 2, 2).unwrap();
    /// ```
    pub fn new(
        data: Array2<T>,
        indices: Array2<usize>,
        nrows: usize,
        ncols: usize,
    ) -> SparseResult<Self> {
        // Validate dimensions
        if data.nrows() != nrows || indices.nrows() != nrows {
            return Err(SparseError::validation(
                "Data/indices row count must match nrows",
            ));
        }

        if data.ncols() != indices.ncols() {
            return Err(SparseError::validation(
                "Data and indices must have same number of columns",
            ));
        }

        let max_nnz_per_row = data.ncols();

        // Validate indices
        for row in 0..nrows {
            for col in 0..max_nnz_per_row {
                let idx = indices[[row, col]];
                if idx != ncols && idx >= ncols {
                    return Err(SparseError::validation(&format!(
                        "Invalid column index {} >= {}",
                        idx, ncols
                    )));
                }
            }
        }

        Ok(Self {
            data,
            indices,
            nrows,
            ncols,
            max_nnz_per_row,
        })
    }

    /// Creates an ELL matrix from CSR format.
    ///
    /// # Complexity
    ///
    /// O(nnz)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::coo::CooTensor;
    /// use tenrso_sparse::csr::CsrMatrix;
    /// use tenrso_sparse::ell::EllMatrix;
    ///
    /// let mut coo = CooTensor::zeros(vec![2, 2]).unwrap();
    /// coo.push(vec![0, 0], 1.0).unwrap();
    /// coo.push(vec![0, 1], 2.0).unwrap();
    /// coo.push(vec![1, 1], 3.0).unwrap();
    /// let csr = CsrMatrix::from_coo(&coo).unwrap();
    ///
    /// let ell = EllMatrix::from_csr(&csr);
    /// assert_eq!(ell.nrows(), 2);
    /// assert_eq!(ell.max_nnz_per_row(), 2);
    /// ```
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        let nrows = csr.nrows();
        let ncols = csr.ncols();

        // Find maximum nnz per row
        let max_nnz_per_row = (0..nrows)
            .map(|i| csr.row_ptr()[i + 1] - csr.row_ptr()[i])
            .max()
            .unwrap_or(0);

        // Allocate arrays
        let mut data = Array2::zeros((nrows, max_nnz_per_row));
        let mut indices = Array2::from_elem((nrows, max_nnz_per_row), ncols); // sentinel

        // Fill from CSR
        for row in 0..nrows {
            let row_start = csr.row_ptr()[row];
            let row_end = csr.row_ptr()[row + 1];

            for (pos, idx) in (row_start..row_end).enumerate() {
                data[[row, pos]] = csr.values()[idx];
                indices[[row, pos]] = csr.col_indices()[idx];
            }
        }

        Self {
            data,
            indices,
            nrows,
            ncols,
            max_nnz_per_row,
        }
    }

    /// Converts ELL matrix to CSR format.
    ///
    /// # Complexity
    ///
    /// O(nrows × max_nnz_per_row)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::ell::EllMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let data = array![[1.0, 2.0], [3.0, 0.0]];
    /// let indices = array![[0, 1], [1, 2]];
    /// let ell = EllMatrix::new(data, indices, 2, 2).unwrap();
    ///
    /// let csr = ell.to_csr().unwrap();
    /// assert_eq!(csr.nnz(), 3);
    /// ```
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>> {
        let mut row_ptr = vec![0; self.nrows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..self.nrows {
            for col_pos in 0..self.max_nnz_per_row {
                let idx = self.indices[[row, col_pos]];
                if idx < self.ncols {
                    // Valid entry (not sentinel)
                    let val = self.data[[row, col_pos]];
                    if val != T::zero() {
                        col_indices.push(idx);
                        values.push(val);
                    }
                }
            }
            row_ptr[row + 1] = col_indices.len();
        }

        CsrMatrix::new(row_ptr, col_indices, values, (self.nrows, self.ncols))
            .map_err(|e| SparseError::conversion(format!("ELL to CSR failed: {}", e)))
    }

    /// Sparse matrix-vector multiplication: `y = A * x`
    ///
    /// # Complexity
    ///
    /// O(nrows × max_nnz_per_row)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::ell::EllMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let data = array![[1.0, 2.0], [3.0, 0.0]];
    /// let indices = array![[0, 1], [1, 2]];
    /// let ell = EllMatrix::new(data, indices, 2, 2).unwrap();
    ///
    /// let x = array![1.0, 2.0];
    /// let y = ell.spmv(&x).unwrap();
    /// assert_eq!(y[[0]], 5.0); // 1*1 + 2*2
    /// assert_eq!(y[[1]], 6.0); // 3*2
    /// ```
    pub fn spmv(&self, x: &Array1<T>) -> SparseResult<Array1<T>> {
        if x.len() != self.ncols {
            return Err(SparseError::validation(&format!(
                "Vector length {} != ncols {}",
                x.len(),
                self.ncols
            )));
        }

        let mut y = Array1::zeros(self.nrows);

        for row in 0..self.nrows {
            let mut sum = T::zero();
            for col_pos in 0..self.max_nnz_per_row {
                let col = self.indices[[row, col_pos]];
                if col < self.ncols {
                    sum = sum + self.data[[row, col_pos]] * x[col];
                }
            }
            y[row] = sum;
        }

        Ok(y)
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the maximum number of non-zeros per row.
    pub fn max_nnz_per_row(&self) -> usize {
        self.max_nnz_per_row
    }

    /// Returns the actual number of non-zeros (excluding padding).
    ///
    /// # Complexity
    ///
    /// O(nrows × max_nnz_per_row)
    pub fn nnz(&self) -> usize {
        let mut count = 0;
        for row in 0..self.nrows {
            for col_pos in 0..self.max_nnz_per_row {
                let col = self.indices[[row, col_pos]];
                if col < self.ncols && self.data[[row, col_pos]] != T::zero() {
                    count += 1;
                }
            }
        }
        count
    }

    /// Returns the shape as `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns a reference to the data array.
    pub fn data(&self) -> &Array2<T> {
        &self.data
    }

    /// Returns a reference to the indices array.
    pub fn indices(&self) -> &Array2<usize> {
        &self.indices
    }

    /// Computes the density of the matrix (nnz / (nrows × ncols)).
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Computes the fill efficiency (nnz / (nrows × max_nnz_per_row)).
    ///
    /// Low efficiency indicates wasted padding space.
    pub fn fill_efficiency(&self) -> f64 {
        let storage_size = self.nrows * self.max_nnz_per_row;
        if storage_size == 0 {
            1.0
        } else {
            self.nnz() as f64 / storage_size as f64
        }
    }
}

impl<T: Float + fmt::Display> fmt::Display for EllMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ELL Matrix {}×{}", self.nrows, self.ncols)?;
        writeln!(
            f,
            "Max NNZ per row: {}, Total NNZ: {}",
            self.max_nnz_per_row,
            self.nnz()
        )?;
        writeln!(f, "Fill efficiency: {:.2}%", self.fill_efficiency() * 100.0)?;

        if self.nrows <= 10 {
            writeln!(f, "\nData:")?;
            for row in 0..self.nrows {
                write!(f, "Row {}: [", row)?;
                for col_pos in 0..self.max_nnz_per_row {
                    if col_pos > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.data[[row, col_pos]])?;
                }
                writeln!(f, "]")?;
            }

            writeln!(f, "\nIndices:")?;
            for row in 0..self.nrows {
                write!(f, "Row {}: [", row)?;
                for col_pos in 0..self.max_nnz_per_row {
                    if col_pos > 0 {
                        write!(f, ", ")?;
                    }
                    let idx = self.indices[[row, col_pos]];
                    if idx == self.ncols {
                        write!(f, "*")?; // sentinel
                    } else {
                        write!(f, "{}", idx)?;
                    }
                }
                writeln!(f, "]")?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooTensor;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_ell_creation() {
        let data = array![[1.0, 2.0], [3.0, 0.0], [4.0, 5.0]];
        let indices = array![[0, 1], [1, 3], [0, 2]];

        let ell = EllMatrix::new(data, indices, 3, 3).unwrap();
        assert_eq!(ell.nrows(), 3);
        assert_eq!(ell.ncols(), 3);
        assert_eq!(ell.max_nnz_per_row(), 2);
    }

    #[test]
    fn test_ell_invalid_indices() {
        let data = array![[1.0, 2.0]];
        let indices = array![[0, 5]]; // 5 is out of bounds for ncols=3
        let result = EllMatrix::new(data, indices, 1, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_ell_from_csr() {
        let mut coo = CooTensor::zeros(vec![3, 2]).unwrap();
        coo.push(vec![0, 0], 1.0).unwrap();
        coo.push(vec![0, 1], 2.0).unwrap();
        coo.push(vec![1, 1], 3.0).unwrap();
        coo.push(vec![2, 0], 4.0).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let ell = EllMatrix::from_csr(&csr);
        assert_eq!(ell.nrows(), 3);
        assert_eq!(ell.ncols(), 2);
        assert_eq!(ell.max_nnz_per_row(), 2);
        assert_eq!(ell.nnz(), 4);
    }

    #[test]
    fn test_ell_to_csr_roundtrip() {
        let mut coo = CooTensor::zeros(vec![2, 2]).unwrap();
        coo.push(vec![0, 0], 1.0).unwrap();
        coo.push(vec![0, 1], 2.0).unwrap();
        coo.push(vec![1, 1], 3.0).unwrap();
        let csr1 = CsrMatrix::from_coo(&coo).unwrap();

        let ell = EllMatrix::from_csr(&csr1);
        let csr2 = ell.to_csr().unwrap();

        assert_eq!(csr1.nrows(), csr2.nrows());
        assert_eq!(csr1.ncols(), csr2.ncols());
        assert_eq!(csr1.nnz(), csr2.nnz());
    }

    #[test]
    fn test_ell_spmv() {
        let data = array![[1.0, 2.0], [3.0, 0.0]];
        let indices = array![[0, 1], [1, 2]];
        let ell = EllMatrix::new(data, indices, 2, 2).unwrap();

        let x = array![1.0, 2.0];
        let y = ell.spmv(&x).unwrap();

        assert_eq!(y[[0]], 5.0); // 1*1 + 2*2
        assert_eq!(y[[1]], 6.0); // 3*2
    }

    #[test]
    fn test_ell_spmv_shape_mismatch() {
        let data = array![[1.0]];
        let indices = array![[0]];
        let ell = EllMatrix::new(data, indices, 1, 2).unwrap();

        let x = array![1.0, 2.0, 3.0]; // Wrong size
        let result = ell.spmv(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_ell_nnz_with_zeros() {
        let data = array![[1.0, 0.0], [0.0, 3.0]];
        let indices = array![[0, 1], [0, 1]];
        let ell = EllMatrix::new(data, indices, 2, 2).unwrap();

        assert_eq!(ell.nnz(), 2); // Only count non-zeros
    }

    #[test]
    fn test_ell_density() {
        let data = array![[1.0, 2.0], [3.0, 0.0]];
        let indices = array![[0, 1], [1, 2]];
        let ell = EllMatrix::new(data, indices, 2, 2).unwrap();

        let density = ell.density();
        assert!(density > 0.0 && density <= 1.0);
    }

    #[test]
    fn test_ell_fill_efficiency() {
        // Uniform rows
        let data1 = array![[1.0, 2.0], [3.0, 4.0]];
        let indices1 = array![[0, 1], [0, 1]];
        let ell1 = EllMatrix::new(data1, indices1, 2, 2).unwrap();
        assert_eq!(ell1.fill_efficiency(), 1.0); // Perfect efficiency

        // Non-uniform rows (with padding)
        let data2 = array![[1.0, 2.0], [3.0, 0.0]];
        let indices2 = array![[0, 1], [0, 2]]; // Second row has padding
        let ell2 = EllMatrix::new(data2, indices2, 2, 2).unwrap();
        assert!(ell2.fill_efficiency() < 1.0); // Some waste
    }

    #[test]
    fn test_ell_empty_matrix() {
        let data = Array2::<f64>::zeros((0, 0));
        let indices = Array2::<usize>::zeros((0, 0));
        let ell = EllMatrix::new(data, indices, 0, 0).unwrap();

        assert_eq!(ell.nrows(), 0);
        assert_eq!(ell.ncols(), 0);
        assert_eq!(ell.nnz(), 0);
    }

    #[test]
    fn test_ell_identity_matrix() {
        let n = 3;
        let mut data = Array2::zeros((n, 1));
        let mut indices = Array2::from_elem((n, 1), n); // sentinel

        for i in 0..n {
            data[[i, 0]] = 1.0;
            indices[[i, 0]] = i;
        }

        let ell = EllMatrix::new(data, indices, n, n).unwrap();
        assert_eq!(ell.nnz(), n);

        let x = array![1.0, 2.0, 3.0];
        let y = ell.spmv(&x).unwrap();
        assert_eq!(y, x); // Identity: I * x = x
    }
}
