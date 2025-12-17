//! Sparse tensor indexing and slicing operations
//!
//! This module provides efficient indexing and slicing for sparse tensors,
//! including element access, row/column extraction, and sub-tensor slicing.
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::{CsrMatrix, indexing::{SparseIndex, SparseRowSlice}};
//! use tenrso_core::DenseND;
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create a sparse matrix
//! let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
//! let dense = DenseND::from_array(arr.into_dyn());
//! let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();
//!
//! // Get element at (1, 1)
//! assert_eq!(csr.get_element(1, 1), Some(&3.0));
//! assert_eq!(csr.get_element(0, 1), None);
//!
//! // Extract row as sparse vector
//! let row_vals = csr.get_row_values(0);
//! assert_eq!(row_vals, &[1.0, 2.0]);
//! ```
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.

use crate::{CooTensor, CscMatrix, CsrMatrix, SparseError, SparseResult};
use scirs2_core::ndarray_ext::Array1;

/// Trait for sparse matrix element access
pub trait SparseIndex<T> {
    /// Get element at (row, col), returns None if zero
    fn get_element(&self, row: usize, col: usize) -> Option<&T>;

    /// Check if element at (row, col) is non-zero
    fn is_nonzero(&self, row: usize, col: usize) -> bool {
        self.get_element(row, col).is_some()
    }

    /// Get element at (row, col), returns zero if not present
    fn get_or_zero(&self, row: usize, col: usize) -> T
    where
        T: Clone + Default,
    {
        self.get_element(row, col).cloned().unwrap_or_default()
    }
}

/// Trait for sparse matrix row slicing
pub trait SparseRowSlice<T> {
    /// Get non-zero values in a row
    fn get_row_values(&self, row: usize) -> &[T];

    /// Get column indices for non-zero values in a row
    fn get_row_indices(&self, row: usize) -> &[usize];

    /// Extract a row as a sparse vector (indices, values)
    fn extract_row(&self, row: usize) -> SparseResult<(Vec<usize>, Vec<T>)>
    where
        T: Clone;

    /// Extract a row as a dense array
    fn extract_row_dense(&self, row: usize) -> SparseResult<Array1<T>>
    where
        T: Clone + Default;
}

/// Trait for sparse matrix column slicing
pub trait SparseColSlice<T> {
    /// Extract a column as a sparse vector (indices, values)
    fn extract_col(&self, col: usize) -> SparseResult<(Vec<usize>, Vec<T>)>
    where
        T: Clone;

    /// Extract a column as a dense array
    fn extract_col_dense(&self, col: usize) -> SparseResult<Array1<T>>
    where
        T: Clone + Default;
}

/// Trait for sparse matrix sub-matrix extraction
pub trait SparseSlice<T> {
    /// Extract a rectangular sub-matrix as COO
    ///
    /// # Arguments
    ///
    /// * `row_range` - Range of rows (start, end)
    /// * `col_range` - Range of columns (start, end)
    ///
    /// # Returns
    ///
    /// COO tensor with elements in the specified range
    fn extract_submatrix(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<CooTensor<T>>
    where
        T: Clone;
}

// ============================================================================
// CSR Matrix Implementations
// ============================================================================

impl<T: Clone> SparseIndex<T> for CsrMatrix<T> {
    fn get_element(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows() || col >= self.ncols() {
            return None;
        }

        let start = self.row_ptr()[row];
        let end = self.row_ptr()[row + 1];
        let col_indices = &self.col_indices()[start..end];
        let values = &self.values()[start..end];

        // Binary search for the column index
        match col_indices.binary_search(&col) {
            Ok(pos) => Some(&values[pos]),
            Err(_) => None,
        }
    }
}

impl<T: Clone> SparseRowSlice<T> for CsrMatrix<T> {
    fn get_row_values(&self, row: usize) -> &[T] {
        if row >= self.nrows() {
            return &[];
        }
        let start = self.row_ptr()[row];
        let end = self.row_ptr()[row + 1];
        &self.values()[start..end]
    }

    fn get_row_indices(&self, row: usize) -> &[usize] {
        if row >= self.nrows() {
            return &[];
        }
        let start = self.row_ptr()[row];
        let end = self.row_ptr()[row + 1];
        &self.col_indices()[start..end]
    }

    fn extract_row(&self, row: usize) -> SparseResult<(Vec<usize>, Vec<T>)> {
        if row >= self.nrows() {
            return Err(SparseError::index_out_of_bounds(
                vec![row],
                vec![self.nrows()],
            ));
        }

        let indices = self.get_row_indices(row).to_vec();
        let values = self.get_row_values(row).to_vec();
        Ok((indices, values))
    }

    fn extract_row_dense(&self, row: usize) -> SparseResult<Array1<T>>
    where
        T: Default,
    {
        if row >= self.nrows() {
            return Err(SparseError::index_out_of_bounds(
                vec![row],
                vec![self.nrows()],
            ));
        }

        let mut dense = Array1::default(self.ncols());
        let start = self.row_ptr()[row];
        let end = self.row_ptr()[row + 1];

        for i in start..end {
            dense[self.col_indices()[i]] = self.values()[i].clone();
        }

        Ok(dense)
    }
}

impl<T: Clone> SparseColSlice<T> for CsrMatrix<T> {
    fn extract_col(&self, col: usize) -> SparseResult<(Vec<usize>, Vec<T>)> {
        if col >= self.ncols() {
            return Err(SparseError::index_out_of_bounds(
                vec![col],
                vec![self.ncols()],
            ));
        }

        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        // Scan all rows for the column
        for row in 0..self.nrows() {
            if let Some(val) = self.get_element(row, col) {
                row_indices.push(row);
                values.push(val.clone());
            }
        }

        Ok((row_indices, values))
    }

    fn extract_col_dense(&self, col: usize) -> SparseResult<Array1<T>>
    where
        T: Default,
    {
        if col >= self.ncols() {
            return Err(SparseError::index_out_of_bounds(
                vec![col],
                vec![self.ncols()],
            ));
        }

        let mut dense = Array1::default(self.nrows());

        for row in 0..self.nrows() {
            if let Some(val) = self.get_element(row, col) {
                dense[row] = val.clone();
            }
        }

        Ok(dense)
    }
}

impl<T: Clone> SparseSlice<T> for CsrMatrix<T> {
    fn extract_submatrix(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<CooTensor<T>> {
        let (row_start, row_end) = row_range;
        let (col_start, col_end) = col_range;

        if row_end > self.nrows() || col_end > self.ncols() {
            return Err(SparseError::index_out_of_bounds(
                vec![row_end, col_end],
                vec![self.nrows(), self.ncols()],
            ));
        }

        if row_start >= row_end || col_start >= col_end {
            return Err(SparseError::validation(
                "Invalid range: start must be less than end",
            ));
        }

        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Scan rows in the range
        for row in row_start..row_end {
            let start = self.row_ptr()[row];
            let end = self.row_ptr()[row + 1];

            for i in start..end {
                let col = self.col_indices()[i];
                if col >= col_start && col < col_end {
                    // Adjust to local coordinates
                    indices.push(vec![row - row_start, col - col_start]);
                    values.push(self.values()[i].clone());
                }
            }
        }

        let shape = vec![row_end - row_start, col_end - col_start];
        CooTensor::new(indices, values, shape).map_err(|e| SparseError::conversion(e.to_string()))
    }
}

// ============================================================================
// CSC Matrix Implementations
// ============================================================================

impl<T: Clone> SparseIndex<T> for CscMatrix<T> {
    fn get_element(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows() || col >= self.ncols() {
            return None;
        }

        let start = self.col_ptr()[col];
        let end = self.col_ptr()[col + 1];
        let row_indices = &self.row_indices()[start..end];
        let values = &self.values()[start..end];

        // Binary search for the row index
        match row_indices.binary_search(&row) {
            Ok(pos) => Some(&values[pos]),
            Err(_) => None,
        }
    }
}

impl<T: Clone> SparseColSlice<T> for CscMatrix<T> {
    fn extract_col(&self, col: usize) -> SparseResult<(Vec<usize>, Vec<T>)> {
        if col >= self.ncols() {
            return Err(SparseError::index_out_of_bounds(
                vec![col],
                vec![self.ncols()],
            ));
        }

        let start = self.col_ptr()[col];
        let end = self.col_ptr()[col + 1];

        let indices = self.row_indices()[start..end].to_vec();
        let values = self.values()[start..end].to_vec();

        Ok((indices, values))
    }

    fn extract_col_dense(&self, col: usize) -> SparseResult<Array1<T>>
    where
        T: Default,
    {
        if col >= self.ncols() {
            return Err(SparseError::index_out_of_bounds(
                vec![col],
                vec![self.ncols()],
            ));
        }

        let mut dense = Array1::default(self.nrows());
        let start = self.col_ptr()[col];
        let end = self.col_ptr()[col + 1];

        for i in start..end {
            dense[self.row_indices()[i]] = self.values()[i].clone();
        }

        Ok(dense)
    }
}

impl<T: Clone> SparseRowSlice<T> for CscMatrix<T> {
    fn get_row_values(&self, _row: usize) -> &[T] {
        // CSC doesn't have efficient row access, return empty
        &[]
    }

    fn get_row_indices(&self, _row: usize) -> &[usize] {
        &[]
    }

    fn extract_row(&self, row: usize) -> SparseResult<(Vec<usize>, Vec<T>)> {
        if row >= self.nrows() {
            return Err(SparseError::index_out_of_bounds(
                vec![row],
                vec![self.nrows()],
            ));
        }

        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Scan all columns for the row
        for col in 0..self.ncols() {
            if let Some(val) = self.get_element(row, col) {
                col_indices.push(col);
                values.push(val.clone());
            }
        }

        Ok((col_indices, values))
    }

    fn extract_row_dense(&self, row: usize) -> SparseResult<Array1<T>>
    where
        T: Default,
    {
        if row >= self.nrows() {
            return Err(SparseError::index_out_of_bounds(
                vec![row],
                vec![self.nrows()],
            ));
        }

        let mut dense = Array1::default(self.ncols());

        for col in 0..self.ncols() {
            if let Some(val) = self.get_element(row, col) {
                dense[col] = val.clone();
            }
        }

        Ok(dense)
    }
}

impl<T: Clone> SparseSlice<T> for CscMatrix<T> {
    fn extract_submatrix(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<CooTensor<T>> {
        let (row_start, row_end) = row_range;
        let (col_start, col_end) = col_range;

        if row_end > self.nrows() || col_end > self.ncols() {
            return Err(SparseError::index_out_of_bounds(
                vec![row_end, col_end],
                vec![self.nrows(), self.ncols()],
            ));
        }

        if row_start >= row_end || col_start >= col_end {
            return Err(SparseError::validation(
                "Invalid range: start must be less than end",
            ));
        }

        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Scan columns in the range
        for col in col_start..col_end {
            let start = self.col_ptr()[col];
            let end = self.col_ptr()[col + 1];

            for i in start..end {
                let row = self.row_indices()[i];
                if row >= row_start && row < row_end {
                    // Adjust to local coordinates
                    indices.push(vec![row - row_start, col - col_start]);
                    values.push(self.values()[i].clone());
                }
            }
        }

        let shape = vec![row_end - row_start, col_end - col_start];
        CooTensor::new(indices, values, shape).map_err(|e| SparseError::conversion(e.to_string()))
    }
}

// ============================================================================
// COO Tensor Implementations
// ============================================================================

impl<T: Clone> SparseIndex<T> for CooTensor<T> {
    fn get_element(&self, row: usize, col: usize) -> Option<&T> {
        if self.shape().len() != 2 {
            return None;
        }

        // Linear search through all elements
        for (idx, indices) in self.indices().iter().enumerate() {
            if indices.len() == 2 && indices[0] == row && indices[1] == col {
                return Some(&self.values()[idx]);
            }
        }

        None
    }
}

impl<T: Clone> CooTensor<T> {
    /// Get element at arbitrary N-dimensional coordinates
    ///
    /// # Complexity
    /// O(nnz) - linear search through all non-zero elements
    pub fn get_at(&self, coords: &[usize]) -> Option<&T> {
        if coords.len() != self.shape().len() {
            return None;
        }

        for (idx, indices) in self.indices().iter().enumerate() {
            if indices == coords {
                return Some(&self.values()[idx]);
            }
        }

        None
    }

    /// Extract elements along a specific axis at a given index
    ///
    /// Returns a COO tensor with reduced dimensionality
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to slice along
    /// * `index` - The index along that axis
    ///
    /// # Complexity
    /// O(nnz) - scans all non-zero elements
    pub fn slice_axis(&self, axis: usize, index: usize) -> SparseResult<CooTensor<T>> {
        if axis >= self.shape().len() {
            return Err(SparseError::validation(&format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.shape().len()
            )));
        }

        if index >= self.shape()[axis] {
            return Err(SparseError::index_out_of_bounds(
                vec![index],
                vec![self.shape()[axis]],
            ));
        }

        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        // Filter elements and remove the sliced axis
        for (idx, coords) in self.indices().iter().enumerate() {
            if coords[axis] == index {
                let mut new_coord = coords.clone();
                new_coord.remove(axis);
                new_indices.push(new_coord);
                new_values.push(self.values()[idx].clone());
            }
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.remove(axis);

        if new_shape.is_empty() {
            new_shape.push(1);
        }

        CooTensor::new(new_indices, new_values, new_shape)
            .map_err(|e| SparseError::conversion(e.to_string()))
    }

    /// Extract a hyper-rectangular sub-tensor
    ///
    /// # Arguments
    ///
    /// * `ranges` - Vec of (start, end) for each dimension
    ///
    /// # Complexity
    /// O(nnz) - scans all non-zero elements
    pub fn extract_box(&self, ranges: &[(usize, usize)]) -> SparseResult<CooTensor<T>> {
        if ranges.len() != self.shape().len() {
            return Err(SparseError::validation(&format!(
                "Range count {} doesn't match tensor dimensions {}",
                ranges.len(),
                self.shape().len()
            )));
        }

        // Validate ranges
        for (dim, &(start, end)) in ranges.iter().enumerate() {
            if end > self.shape()[dim] || start >= end {
                return Err(SparseError::validation(&format!(
                    "Invalid range [{}, {}) for dimension {} with size {}",
                    start,
                    end,
                    dim,
                    self.shape()[dim]
                )));
            }
        }

        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        // Filter elements within the box
        for (idx, coords) in self.indices().iter().enumerate() {
            let mut in_box = true;
            for (dim, &coord) in coords.iter().enumerate() {
                let (start, end) = ranges[dim];
                if coord < start || coord >= end {
                    in_box = false;
                    break;
                }
            }

            if in_box {
                // Adjust to local coordinates
                let local_coords: Vec<usize> = coords
                    .iter()
                    .enumerate()
                    .map(|(dim, &c)| c - ranges[dim].0)
                    .collect();
                new_indices.push(local_coords);
                new_values.push(self.values()[idx].clone());
            }
        }

        let new_shape: Vec<usize> = ranges.iter().map(|(start, end)| end - start).collect();

        CooTensor::new(new_indices, new_values, new_shape)
            .map_err(|e| SparseError::conversion(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;
    use tenrso_core::DenseND;

    #[test]
    fn test_csr_get_element() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        assert_eq!(csr.get_element(0, 0), Some(&1.0));
        assert_eq!(csr.get_element(1, 1), Some(&3.0));
        assert_eq!(csr.get_element(2, 2), Some(&5.0));
        assert_eq!(csr.get_element(0, 1), None);
        assert_eq!(csr.get_element(1, 0), None);
    }

    #[test]
    fn test_csr_extract_row() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let (indices, values) = csr.extract_row(0).unwrap();
        assert_eq!(indices, vec![0, 2]);
        assert_eq!(values, vec![1.0, 2.0]);

        let (indices, values) = csr.extract_row(1).unwrap();
        assert_eq!(indices, vec![1]);
        assert_eq!(values, vec![3.0]);
    }

    #[test]
    fn test_csr_extract_row_dense() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let row = csr.extract_row_dense(0).unwrap();
        assert_eq!(row, array![1.0, 0.0, 2.0]);

        let row = csr.extract_row_dense(1).unwrap();
        assert_eq!(row, array![0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_csr_extract_col() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let (indices, values) = csr.extract_col(0).unwrap();
        assert_eq!(indices, vec![0, 2]);
        assert_eq!(values, vec![1.0, 4.0]);

        let (indices, values) = csr.extract_col(1).unwrap();
        assert_eq!(indices, vec![1]);
        assert_eq!(values, vec![3.0]);
    }

    #[test]
    fn test_csr_extract_submatrix() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let sub = csr.extract_submatrix((0, 2), (0, 2)).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        assert_eq!(sub.nnz(), 2); // 1.0 and 3.0

        let sub = csr.extract_submatrix((1, 3), (1, 3)).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        assert_eq!(sub.nnz(), 2); // 3.0 and 5.0
    }

    #[test]
    fn test_csc_get_element() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 1e-10).unwrap();

        assert_eq!(csc.get_element(0, 0), Some(&1.0));
        assert_eq!(csc.get_element(1, 1), Some(&3.0));
        assert_eq!(csc.get_element(2, 2), Some(&5.0));
        assert_eq!(csc.get_element(0, 1), None);
    }

    #[test]
    fn test_csc_extract_col() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 1e-10).unwrap();

        let (indices, values) = csc.extract_col(0).unwrap();
        assert_eq!(indices, vec![0, 2]);
        assert_eq!(values, vec![1.0, 4.0]);
    }

    #[test]
    fn test_coo_get_at() {
        let indices = vec![vec![0, 0], vec![1, 1], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        assert_eq!(coo.get_at(&[0, 0]), Some(&1.0));
        assert_eq!(coo.get_at(&[1, 1]), Some(&2.0));
        assert_eq!(coo.get_at(&[0, 1]), None);
    }

    #[test]
    fn test_coo_slice_axis() {
        let indices = vec![vec![0, 0, 1], vec![1, 1, 2], vec![0, 1, 1], vec![1, 0, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2, 3];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        // Slice along axis 0 at index 0
        let sliced = coo.slice_axis(0, 0).unwrap();
        assert_eq!(sliced.shape(), &[2, 3]);
        assert_eq!(sliced.nnz(), 2); // (0,1) and (1,1)
    }

    #[test]
    fn test_coo_extract_box() {
        let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        // Extract top-left 2x2 sub-matrix
        let sub = coo.extract_box(&[(0, 2), (0, 2)]).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        assert_eq!(sub.nnz(), 2); // (0,0)=1.0 and (1,1)=3.0
    }

    #[test]
    fn test_sparse_index_trait() {
        let arr = array![[1.0, 0.0], [0.0, 2.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        assert!(csr.is_nonzero(0, 0));
        assert!(!csr.is_nonzero(0, 1));
        assert_eq!(csr.get_or_zero(0, 1), 0.0);
        assert_eq!(csr.get_or_zero(1, 1), 2.0);
    }
}
