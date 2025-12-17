//! Efficient iterators for sparse tensor traversal
//!
//! This module provides zero-copy iterators for traversing sparse tensors
//! in various patterns optimized for each format.
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::{CsrMatrix, iterators::SparseIterator};
//! use tenrso_core::DenseND;
//! use scirs2_core::ndarray_ext::array;
//!
//! let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
//! let dense = DenseND::from_array(arr.into_dyn());
//! let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();
//!
//! // Iterate over all non-zero elements
//! for (row, col, &value) in csr.iter_nonzero() {
//!     println!("({}, {}) = {}", row, col, value);
//! }
//! ```
//!
//! # SciRS2 Integration
//!
//! All operations use `scirs2_core` types.

use crate::{CooTensor, CscMatrix, CsrMatrix};

// ============================================================================
// Core Iterator Traits
// ============================================================================

/// Trait for iterating over non-zero elements in a sparse matrix
pub trait SparseIterator<T> {
    /// Iterator type that yields (row, col, value) tuples
    type Iter<'a>: Iterator<Item = (usize, usize, &'a T)>
    where
        T: 'a,
        Self: 'a;

    /// Create an iterator over all non-zero elements
    ///
    /// # Complexity
    /// - Iteration: O(nnz)
    /// - Memory: O(1) per step
    fn iter_nonzero(&self) -> Self::Iter<'_>;
}

/// Trait for iterating over rows in a sparse matrix
pub trait RowIterator<T> {
    /// Row view type
    type RowView<'a>: Iterator<Item = (usize, &'a T)>
    where
        T: 'a,
        Self: 'a;

    /// Iterator over rows
    type Rows<'a>: Iterator<Item = (usize, Self::RowView<'a>)>
    where
        T: 'a,
        Self: 'a;

    /// Iterate over all rows
    ///
    /// Returns an iterator of (row_index, row_elements) where row_elements
    /// is an iterator over (col_index, value) pairs.
    fn iter_rows(&self) -> Self::Rows<'_>;

    /// Get an iterator for a specific row
    fn row_iter(&self, row: usize) -> Option<Self::RowView<'_>>;
}

/// Trait for iterating over columns in a sparse matrix
pub trait ColIterator<T> {
    /// Column view type
    type ColView<'a>: Iterator<Item = (usize, &'a T)>
    where
        T: 'a,
        Self: 'a;

    /// Iterator over columns
    type Cols<'a>: Iterator<Item = (usize, Self::ColView<'a>)>
    where
        T: 'a,
        Self: 'a;

    /// Iterate over all columns
    fn iter_cols(&self) -> Self::Cols<'_>;

    /// Get an iterator for a specific column
    fn col_iter(&self, col: usize) -> Option<Self::ColView<'_>>;
}

// ============================================================================
// CSR Matrix Iterators
// ============================================================================

/// Iterator over non-zero elements in CSR matrix
pub struct CsrNonZeroIter<'a, T> {
    row_ptr: &'a [usize],
    col_indices: &'a [usize],
    values: &'a [T],
    current_row: usize,
    current_idx: usize,
}

impl<'a, T> Iterator for CsrNonZeroIter<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_row < self.row_ptr.len() - 1 {
            let row_end = self.row_ptr[self.current_row + 1];

            if self.current_idx < row_end {
                let idx = self.current_idx;
                let col = self.col_indices[idx];
                let val = &self.values[idx];
                self.current_idx += 1;
                return Some((self.current_row, col, val));
            }

            // Move to next row
            self.current_row += 1;
            if self.current_row < self.row_ptr.len() - 1 {
                self.current_idx = self.row_ptr[self.current_row];
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.values.len() - self.current_idx;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for CsrNonZeroIter<'a, T> {}

/// Iterator over a single row in CSR matrix
#[derive(Clone)]
pub struct CsrRowIter<'a, T> {
    col_indices: &'a [usize],
    values: &'a [T],
}

impl<'a, T> Iterator for CsrRowIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.col_indices.is_empty() {
            return None;
        }

        let col = self.col_indices[0];
        let val = &self.values[0];
        self.col_indices = &self.col_indices[1..];
        self.values = &self.values[1..];

        Some((col, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.col_indices.len(), Some(self.col_indices.len()))
    }
}

impl<'a, T> ExactSizeIterator for CsrRowIter<'a, T> {}

/// Iterator over all rows in CSR matrix
pub struct CsrRowsIter<'a, T> {
    matrix: &'a CsrMatrix<T>,
    current_row: usize,
}

impl<'a, T: Clone> Iterator for CsrRowsIter<'a, T> {
    type Item = (usize, CsrRowIter<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.matrix.nrows() {
            return None;
        }

        let row = self.current_row;
        let start = self.matrix.row_ptr()[row];
        let end = self.matrix.row_ptr()[row + 1];

        let iter = CsrRowIter {
            col_indices: &self.matrix.col_indices()[start..end],
            values: &self.matrix.values()[start..end],
        };

        self.current_row += 1;
        Some((row, iter))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.nrows() - self.current_row;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Clone> ExactSizeIterator for CsrRowsIter<'a, T> {}

impl<T: Clone> SparseIterator<T> for CsrMatrix<T> {
    type Iter<'a>
        = CsrNonZeroIter<'a, T>
    where
        T: 'a;

    fn iter_nonzero(&self) -> Self::Iter<'_> {
        CsrNonZeroIter {
            row_ptr: self.row_ptr(),
            col_indices: self.col_indices(),
            values: self.values(),
            current_row: 0,
            current_idx: if self.row_ptr().len() > 1 {
                self.row_ptr()[0]
            } else {
                0
            },
        }
    }
}

impl<T: Clone> RowIterator<T> for CsrMatrix<T> {
    type RowView<'a>
        = CsrRowIter<'a, T>
    where
        T: 'a;
    type Rows<'a>
        = CsrRowsIter<'a, T>
    where
        T: 'a;

    fn iter_rows(&self) -> Self::Rows<'_> {
        CsrRowsIter {
            matrix: self,
            current_row: 0,
        }
    }

    fn row_iter(&self, row: usize) -> Option<Self::RowView<'_>> {
        if row >= self.nrows() {
            return None;
        }

        let start = self.row_ptr()[row];
        let end = self.row_ptr()[row + 1];

        Some(CsrRowIter {
            col_indices: &self.col_indices()[start..end],
            values: &self.values()[start..end],
        })
    }
}

// ============================================================================
// CSC Matrix Iterators
// ============================================================================

/// Iterator over non-zero elements in CSC matrix
pub struct CscNonZeroIter<'a, T> {
    col_ptr: &'a [usize],
    row_indices: &'a [usize],
    values: &'a [T],
    current_col: usize,
    current_idx: usize,
}

impl<'a, T> Iterator for CscNonZeroIter<'a, T> {
    type Item = (usize, usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_col < self.col_ptr.len() - 1 {
            let col_end = self.col_ptr[self.current_col + 1];

            if self.current_idx < col_end {
                let idx = self.current_idx;
                let row = self.row_indices[idx];
                let val = &self.values[idx];
                self.current_idx += 1;
                return Some((row, self.current_col, val));
            }

            // Move to next column
            self.current_col += 1;
            if self.current_col < self.col_ptr.len() - 1 {
                self.current_idx = self.col_ptr[self.current_col];
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.values.len() - self.current_idx;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for CscNonZeroIter<'a, T> {}

/// Iterator over a single column in CSC matrix
#[derive(Clone)]
pub struct CscColIter<'a, T> {
    row_indices: &'a [usize],
    values: &'a [T],
}

impl<'a, T> Iterator for CscColIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_indices.is_empty() {
            return None;
        }

        let row = self.row_indices[0];
        let val = &self.values[0];
        self.row_indices = &self.row_indices[1..];
        self.values = &self.values[1..];

        Some((row, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.row_indices.len(), Some(self.row_indices.len()))
    }
}

impl<'a, T> ExactSizeIterator for CscColIter<'a, T> {}

/// Iterator over all columns in CSC matrix
pub struct CscColsIter<'a, T> {
    matrix: &'a CscMatrix<T>,
    current_col: usize,
}

impl<'a, T: Clone> Iterator for CscColsIter<'a, T> {
    type Item = (usize, CscColIter<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.matrix.ncols() {
            return None;
        }

        let col = self.current_col;
        let start = self.matrix.col_ptr()[col];
        let end = self.matrix.col_ptr()[col + 1];

        let iter = CscColIter {
            row_indices: &self.matrix.row_indices()[start..end],
            values: &self.matrix.values()[start..end],
        };

        self.current_col += 1;
        Some((col, iter))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.ncols() - self.current_col;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Clone> ExactSizeIterator for CscColsIter<'a, T> {}

impl<T: Clone> SparseIterator<T> for CscMatrix<T> {
    type Iter<'a>
        = CscNonZeroIter<'a, T>
    where
        T: 'a;

    fn iter_nonzero(&self) -> Self::Iter<'_> {
        CscNonZeroIter {
            col_ptr: self.col_ptr(),
            row_indices: self.row_indices(),
            values: self.values(),
            current_col: 0,
            current_idx: if self.col_ptr().len() > 1 {
                self.col_ptr()[0]
            } else {
                0
            },
        }
    }
}

impl<T: Clone> ColIterator<T> for CscMatrix<T> {
    type ColView<'a>
        = CscColIter<'a, T>
    where
        T: 'a;
    type Cols<'a>
        = CscColsIter<'a, T>
    where
        T: 'a;

    fn iter_cols(&self) -> Self::Cols<'_> {
        CscColsIter {
            matrix: self,
            current_col: 0,
        }
    }

    fn col_iter(&self, col: usize) -> Option<Self::ColView<'_>> {
        if col >= self.ncols() {
            return None;
        }

        let start = self.col_ptr()[col];
        let end = self.col_ptr()[col + 1];

        Some(CscColIter {
            row_indices: &self.row_indices()[start..end],
            values: &self.values()[start..end],
        })
    }
}

// ============================================================================
// COO Tensor Iterators
// ============================================================================

/// Iterator over non-zero elements in COO tensor
pub struct CooNonZeroIter<'a, T> {
    indices: &'a [Vec<usize>],
    values: &'a [T],
    current_idx: usize,
}

impl<'a, T> Iterator for CooNonZeroIter<'a, T> {
    type Item = (&'a [usize], &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let idx = self.current_idx;
        self.current_idx += 1;

        Some((&self.indices[idx], &self.values[idx]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.current_idx;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for CooNonZeroIter<'a, T> {}

impl<T: Clone> CooTensor<T> {
    /// Iterate over all non-zero elements
    ///
    /// Returns an iterator yielding (coordinates, value) pairs.
    ///
    /// # Complexity
    /// O(nnz) to iterate all elements
    pub fn iter_nonzero(&self) -> CooNonZeroIter<'_, T> {
        CooNonZeroIter {
            indices: self.indices(),
            values: self.values(),
            current_idx: 0,
        }
    }

    /// Iterate over all non-zero elements with enumeration
    ///
    /// Returns an iterator yielding (index, coordinates, value) tuples.
    pub fn iter_nonzero_enumerated(&self) -> impl Iterator<Item = (usize, &[usize], &T)> {
        self.indices()
            .iter()
            .zip(self.values())
            .enumerate()
            .map(|(i, (coords, val))| (i, coords.as_slice(), val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;
    use tenrso_core::DenseND;

    #[test]
    fn test_csr_nonzero_iter() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let elements: Vec<_> = csr.iter_nonzero().collect();

        assert_eq!(elements.len(), 5);
        assert_eq!(elements[0], (0, 0, &1.0));
        assert_eq!(elements[1], (0, 2, &2.0));
        assert_eq!(elements[2], (1, 1, &3.0));
        assert_eq!(elements[3], (2, 0, &4.0));
        assert_eq!(elements[4], (2, 2, &5.0));
    }

    #[test]
    fn test_csr_row_iter() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let row0: Vec<_> = csr.row_iter(0).unwrap().collect();
        assert_eq!(row0, vec![(0, &1.0), (2, &2.0)]);

        let row1: Vec<_> = csr.row_iter(1).unwrap().collect();
        assert_eq!(row1, vec![(1, &3.0)]);

        assert!(csr.row_iter(2).is_none());
    }

    #[test]
    fn test_csr_rows_iter() {
        let arr = array![[1.0, 0.0], [2.0, 3.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let rows: Vec<_> = csr
            .iter_rows()
            .map(|(idx, row)| (idx, row.collect::<Vec<_>>()))
            .collect();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, 0);
        assert_eq!(rows[0].1, vec![(0, &1.0)]);
        assert_eq!(rows[1].0, 1);
        assert_eq!(rows[1].1, vec![(0, &2.0), (1, &3.0)]);
    }

    #[test]
    fn test_csc_nonzero_iter() {
        let arr = array![[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 1e-10).unwrap();

        let elements: Vec<_> = csc.iter_nonzero().collect();

        assert_eq!(elements.len(), 3);
        assert_eq!(elements[0], (0, 0, &1.0));
        assert_eq!(elements[1], (2, 0, &3.0));
        assert_eq!(elements[2], (1, 1, &2.0));
    }

    #[test]
    fn test_csc_col_iter() {
        let arr = array![[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 1e-10).unwrap();

        let col0: Vec<_> = csc.col_iter(0).unwrap().collect();
        assert_eq!(col0, vec![(0, &1.0), (2, &3.0)]);

        let col1: Vec<_> = csc.col_iter(1).unwrap().collect();
        assert_eq!(col1, vec![(1, &2.0)]);

        assert!(csc.col_iter(2).is_none());
    }

    #[test]
    fn test_csc_cols_iter() {
        let arr = array![[1.0, 2.0], [3.0, 0.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csc = CscMatrix::from_dense(&dense, 1e-10).unwrap();

        let cols: Vec<_> = csc
            .iter_cols()
            .map(|(idx, col)| (idx, col.collect::<Vec<_>>()))
            .collect();

        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].0, 0);
        assert_eq!(cols[0].1, vec![(0, &1.0), (1, &3.0)]);
        assert_eq!(cols[1].0, 1);
        assert_eq!(cols[1].1, vec![(0, &2.0)]);
    }

    #[test]
    fn test_coo_nonzero_iter() {
        let indices = vec![vec![0, 0], vec![1, 1], vec![2, 0]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 2];
        let coo = CooTensor::new(indices, values, shape).unwrap();

        let elements: Vec<_> = coo.iter_nonzero().collect();

        assert_eq!(elements.len(), 3);
        assert_eq!(elements[0].0, &[0, 0]);
        assert_eq!(elements[0].1, &1.0);
        assert_eq!(elements[1].0, &[1, 1]);
        assert_eq!(elements[1].1, &2.0);
        assert_eq!(elements[2].0, &[2, 0]);
        assert_eq!(elements[2].1, &3.0);
    }

    #[test]
    fn test_iterator_size_hints() {
        let arr = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let dense = DenseND::from_array(arr.into_dyn());
        let csr = CsrMatrix::from_dense(&dense, 1e-10).unwrap();

        let iter = csr.iter_nonzero();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.len(), 3);

        let row_iter = csr.row_iter(0).unwrap();
        assert_eq!(row_iter.size_hint(), (2, Some(2)));
        assert_eq!(row_iter.len(), 2);
    }

    #[test]
    fn test_empty_iterators() {
        let csr = CsrMatrix::<f64>::new(vec![0, 0, 0], vec![], vec![], (2, 3)).unwrap();

        assert_eq!(csr.iter_nonzero().count(), 0);
        assert_eq!(csr.iter_rows().count(), 2);

        for (_, row) in csr.iter_rows() {
            assert_eq!(row.count(), 0);
        }
    }
}
