//! DIA (Diagonal) Sparse Matrix Format
//!
//! The Diagonal format stores sparse matrices as a set of diagonals.
//! This format is highly efficient for banded matrices, which are common in
//! PDE solvers, finite difference/element methods, and image processing.
//!
//! # Structure
//!
//! - `data`: 2D array of diagonal values `(num_diagonals × n)`
//! - `offsets`: Array of diagonal offsets (0 = main, +k = kth super, -k = kth sub)
//! - Out-of-bounds entries in data are ignored
//!
//! # Use Cases
//!
//! - Banded matrices from PDEs
//! - Tri-diagonal systems
//! - Pentadiagonal/heptadiagonal systems
//! - Image convolution kernels
//!
//! # Performance
//!
//! - **SpMV**: O(num_diagonals × n) - very efficient for small bandwidth
//! - **Memory**: O(num_diagonals × n) - optimal for banded matrices
//! - **Best for**: Matrices with few diagonals (bandwidth << n)
//!
//! # Example
//!
//! ```
//! use tenrso_sparse::dia::DiaMatrix;
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create a tridiagonal matrix
//! // [ 2 -1  0  0]
//! // [-1  2 -1  0]
//! // [ 0 -1  2 -1]
//! // [ 0  0 -1  2]
//! let data = array![
//!     [-1.0, -1.0, -1.0, 0.0],   // subdiagonal (offset -1)
//!     [2.0, 2.0, 2.0, 2.0],       // main diagonal (offset 0)
//!     [0.0, -1.0, -1.0, -1.0]     // superdiagonal (offset +1)
//! ];
//! let offsets = vec![-1, 0, 1];
//!
//! let dia = DiaMatrix::new(data, offsets, 4, 4).unwrap();
//! assert_eq!(dia.nrows(), 4);
//! assert_eq!(dia.num_diagonals(), 3);
//! ```

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::fmt;

/// DIA (Diagonal) sparse matrix format.
///
/// Stores sparse matrices as a collection of diagonals.
/// Highly efficient for banded matrices.
#[derive(Debug, Clone)]
pub struct DiaMatrix<T> {
    /// Diagonal data `(num_diagonals × max(nrows, ncols))`
    data: Array2<T>,
    /// Diagonal offsets (0 = main, +k = kth super, -k = kth sub)
    offsets: Vec<isize>,
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
}

impl<T: Float> DiaMatrix<T> {
    /// Creates a new DIA matrix from diagonal data and offsets.
    ///
    /// # Arguments
    ///
    /// - `data`: Diagonal values `(num_diagonals × storage_len)`
    /// - `offsets`: Diagonal offsets (must be unique and sorted)
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are inconsistent or offsets are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::dia::DiaMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let offsets = vec![0, 1]; // main and first superdiagonal
    /// let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();
    /// ```
    pub fn new(
        data: Array2<T>,
        offsets: Vec<isize>,
        nrows: usize,
        ncols: usize,
    ) -> SparseResult<Self> {
        // Validate offsets length matches data rows
        if data.nrows() != offsets.len() {
            return Err(SparseError::validation(
                "Number of diagonals must match number of offsets",
            ));
        }

        // Validate storage length
        let storage_len = nrows.max(ncols);
        if data.ncols() != storage_len {
            return Err(SparseError::validation(&format!(
                "Data storage length {} must equal max(nrows, ncols) = {}",
                data.ncols(),
                storage_len
            )));
        }

        // Validate offsets are within bounds
        for &offset in &offsets {
            if offset < -(nrows as isize) || offset >= ncols as isize {
                return Err(SparseError::validation(&format!(
                    "Offset {} out of bounds for {}×{} matrix",
                    offset, nrows, ncols
                )));
            }
        }

        Ok(Self {
            data,
            offsets,
            nrows,
            ncols,
        })
    }

    /// Creates a DIA matrix from CSR format.
    ///
    /// Automatically detects and extracts diagonals from the CSR matrix.
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
    /// use tenrso_sparse::dia::DiaMatrix;
    ///
    /// let mut coo = CooTensor::zeros(vec![3, 3]).unwrap();
    /// coo.push(vec![0, 0], 2.0).unwrap();
    /// coo.push(vec![0, 1], -1.0).unwrap();
    /// coo.push(vec![1, 1], 2.0).unwrap();
    /// let csr = CsrMatrix::from_coo(&coo).unwrap();
    ///
    /// let dia = DiaMatrix::from_csr(&csr);
    /// assert!(dia.num_diagonals() <= 2);
    /// ```
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        use std::collections::HashMap;

        let nrows = csr.nrows();
        let ncols = csr.ncols();
        let storage_len = nrows.max(ncols);

        // Identify which diagonals have nonzeros
        let mut diagonal_map: HashMap<isize, Vec<(usize, T)>> = HashMap::new();

        for row in 0..nrows {
            let row_start = csr.row_ptr()[row];
            let row_end = csr.row_ptr()[row + 1];

            for idx in row_start..row_end {
                let col = csr.col_indices()[idx];
                let val = csr.values()[idx];

                if val != T::zero() {
                    let offset = col as isize - row as isize;
                    diagonal_map.entry(offset).or_default().push((row, val));
                }
            }
        }

        // Sort offsets
        let mut offsets: Vec<isize> = diagonal_map.keys().copied().collect();
        offsets.sort_unstable();

        // Build data array
        let num_diagonals = offsets.len();
        let mut data = Array2::zeros((num_diagonals, storage_len));

        for (diag_idx, &offset) in offsets.iter().enumerate() {
            if let Some(entries) = diagonal_map.get(&offset) {
                for &(row, val) in entries {
                    let col = (row as isize + offset) as usize;
                    if col < ncols {
                        data[[diag_idx, row]] = val;
                    }
                }
            }
        }

        Self {
            data,
            offsets,
            nrows,
            ncols,
        }
    }

    /// Converts DIA matrix to CSR format.
    ///
    /// # Complexity
    ///
    /// O(num_diagonals × n)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::dia::DiaMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// let data = array![[2.0, 2.0, 2.0], [-1.0, -1.0, 0.0]];
    /// let offsets = vec![0, 1];
    /// let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();
    ///
    /// let csr = dia.to_csr().unwrap();
    /// assert!(csr.nnz() > 0);
    /// ```
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>> {
        let mut row_data: Vec<Vec<(usize, T)>> = vec![Vec::new(); self.nrows];

        // Collect all nonzeros by row
        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            for (row, row_vec) in row_data.iter_mut().enumerate().take(self.nrows) {
                let col = row as isize + offset;
                if col >= 0 && (col as usize) < self.ncols {
                    let val = self.data[[diag_idx, row]];
                    if val != T::zero() {
                        row_vec.push((col as usize, val));
                    }
                }
            }
        }

        // Sort by column within each row
        for row in &mut row_data {
            row.sort_by_key(|(col, _)| *col);
        }

        // Build CSR arrays
        let mut row_ptr = vec![0; self.nrows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (row, entries) in row_data.iter().enumerate() {
            for &(col, val) in entries {
                col_indices.push(col);
                values.push(val);
            }
            row_ptr[row + 1] = col_indices.len();
        }

        CsrMatrix::new(row_ptr, col_indices, values, (self.nrows, self.ncols))
            .map_err(|e| SparseError::conversion(format!("DIA to CSR failed: {}", e)))
    }

    /// Sparse matrix-vector multiplication: `y = A * x`
    ///
    /// # Complexity
    ///
    /// O(num_diagonals × n)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::dia::DiaMatrix;
    /// use scirs2_core::ndarray_ext::array;
    ///
    /// // Tridiagonal matrix
    /// let data = array![
    ///     [-1.0, -1.0, 0.0],
    ///     [2.0, 2.0, 2.0],
    ///     [0.0, -1.0, -1.0]
    /// ];
    /// let offsets = vec![-1, 0, 1];
    /// let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();
    ///
    /// let x = array![1.0, 1.0, 1.0];
    /// let y = dia.spmv(&x).unwrap();
    /// // Result: [1.0, 0.0, 1.0] = [2-1, -1+2-1, -1+2]
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

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            for row in 0..self.nrows {
                let col = row as isize + offset;
                if col >= 0 && (col as usize) < self.ncols {
                    let val = self.data[[diag_idx, row]];
                    y[row] = y[row] + val * x[col as usize];
                }
            }
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

    /// Returns the number of stored diagonals.
    pub fn num_diagonals(&self) -> usize {
        self.offsets.len()
    }

    /// Returns a reference to the diagonal offsets.
    pub fn offsets(&self) -> &[isize] {
        &self.offsets
    }

    /// Returns a reference to the data array.
    pub fn data(&self) -> &Array2<T> {
        &self.data
    }

    /// Returns the actual number of non-zeros.
    ///
    /// # Complexity
    ///
    /// O(num_diagonals × n)
    pub fn nnz(&self) -> usize {
        let mut count = 0;
        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            for row in 0..self.nrows {
                let col = row as isize + offset;
                if col >= 0
                    && (col as usize) < self.ncols
                    && self.data[[diag_idx, row]] != T::zero()
                {
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

    /// Computes the density of the matrix.
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Computes the bandwidth of the matrix.
    ///
    /// Returns `(lower_bandwidth, upper_bandwidth)`.
    pub fn bandwidth(&self) -> (usize, usize) {
        let lower = self
            .offsets
            .iter()
            .filter(|&&o| o < 0)
            .map(|&o| (-o) as usize)
            .max()
            .unwrap_or(0);
        let upper = self
            .offsets
            .iter()
            .filter(|&&o| o > 0)
            .map(|&o| o as usize)
            .max()
            .unwrap_or(0);
        (lower, upper)
    }
}

impl<T: Float + fmt::Display> fmt::Display for DiaMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DIA Matrix {}×{}", self.nrows, self.ncols)?;
        writeln!(
            f,
            "Num diagonals: {}, NNZ: {}",
            self.num_diagonals(),
            self.nnz()
        )?;
        let (lower, upper) = self.bandwidth();
        writeln!(f, "Bandwidth: lower={}, upper={}", lower, upper)?;
        writeln!(f, "Offsets: {:?}", self.offsets)?;

        if self.nrows <= 8 && self.num_diagonals() <= 5 {
            writeln!(f, "\nDiagonal data:")?;
            for (idx, &offset) in self.offsets.iter().enumerate() {
                write!(f, "Diagonal {} (offset {}): [", idx, offset)?;
                for col in 0..self.data.ncols().min(8) {
                    if col > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.data[[idx, col]])?;
                }
                if self.data.ncols() > 8 {
                    write!(f, ", ...")?;
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
    fn test_dia_creation() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let offsets = vec![0, 1];
        let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();

        assert_eq!(dia.nrows(), 3);
        assert_eq!(dia.ncols(), 3);
        assert_eq!(dia.num_diagonals(), 2);
    }

    #[test]
    fn test_dia_tridiagonal() {
        // Tridiagonal matrix
        let data = array![
            [-1.0, -1.0, -1.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
            [0.0, -1.0, -1.0, -1.0]
        ];
        let offsets = vec![-1, 0, 1];
        let dia = DiaMatrix::new(data, offsets, 4, 4).unwrap();

        assert_eq!(dia.num_diagonals(), 3);
        assert_eq!(dia.bandwidth(), (1, 1));
    }

    #[test]
    fn test_dia_invalid_storage_length() {
        let data = array![[1.0, 2.0]]; // Wrong length
        let offsets = vec![0];
        let result = DiaMatrix::new(data, offsets, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_dia_invalid_offset_count() {
        let data = array![[1.0, 2.0, 3.0]];
        let offsets = vec![0, 1]; // Too many offsets
        let result = DiaMatrix::new(data, offsets, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_dia_spmv() {
        // Identity matrix as DIA
        let data = array![[1.0, 1.0, 1.0]];
        let offsets = vec![0];
        let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();

        let x = array![1.0, 2.0, 3.0];
        let y = dia.spmv(&x).unwrap();

        assert_eq!(y, x); // Identity: I * x = x
    }

    #[test]
    fn test_dia_spmv_tridiagonal() {
        let data = array![[-1.0, -1.0, 0.0], [2.0, 2.0, 2.0], [0.0, -1.0, -1.0]];
        let offsets = vec![-1, 0, 1];
        let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();

        let x = array![1.0, 1.0, 1.0];
        let y = dia.spmv(&x).unwrap();

        // Row 0: 2*x[0] + 0 = 2
        // Row 1: -1*x[0] + 2*x[1] + -1*x[2] = -1+2-1 = 0
        // Row 2: 0 + 2*x[2] = 2
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 0.0).abs() < 1e-10);
        assert!((y[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dia_to_csr_roundtrip() {
        let data = array![[2.0, 2.0, 2.0], [-1.0, -1.0, 0.0]];
        let offsets = vec![0, 1];
        let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();

        let csr = dia.to_csr().unwrap();
        let dia2 = DiaMatrix::from_csr(&csr);

        assert_eq!(dia.nrows(), dia2.nrows());
        assert_eq!(dia.ncols(), dia2.ncols());
        assert_eq!(dia.nnz(), dia2.nnz());
    }

    #[test]
    fn test_dia_from_csr() {
        let mut coo = CooTensor::zeros(vec![3, 3]).unwrap();
        coo.push(vec![0, 0], 2.0).unwrap();
        coo.push(vec![0, 1], -1.0).unwrap();
        coo.push(vec![1, 0], -1.0).unwrap();
        coo.push(vec![1, 1], 2.0).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let dia = DiaMatrix::from_csr(&csr);
        // We have 3 diagonals: -1 (subdiag), 0 (main), 1 (superdiag)
        assert_eq!(dia.num_diagonals(), 3);
        assert_eq!(dia.nrows(), 3);
        assert_eq!(dia.ncols(), 3);
    }

    #[test]
    fn test_dia_nnz() {
        let data = array![[1.0, 0.0, 2.0], [3.0, 4.0, 0.0]];
        let offsets = vec![0, 1];
        let dia = DiaMatrix::new(data, offsets, 3, 3).unwrap();

        assert_eq!(dia.nnz(), 4); // Only non-zero values
    }

    #[test]
    fn test_dia_empty_matrix() {
        let data = Array2::<f64>::zeros((0, 0));
        let offsets = vec![];
        let dia = DiaMatrix::new(data, offsets, 0, 0).unwrap();

        assert_eq!(dia.nrows(), 0);
        assert_eq!(dia.ncols(), 0);
        assert_eq!(dia.nnz(), 0);
    }

    #[test]
    fn test_dia_bandwidth() {
        let data = array![
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 4.0, 4.0]
        ];
        let offsets = vec![-2, 0, 1, 2];
        let dia = DiaMatrix::new(data, offsets, 4, 4).unwrap();

        assert_eq!(dia.bandwidth(), (2, 2));
    }
}
