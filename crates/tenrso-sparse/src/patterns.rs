//! # Sparse Matrix Pattern Analysis
//!
//! This module provides functions to analyze structural patterns in sparse matrices,
//! which is crucial for:
//! - Algorithm selection (e.g., choosing iterative solvers)
//! - Preconditioning strategies
//! - Memory layout optimization
//! - Graph analysis
//!
//! ## Available Analyses
//!
//! - **Bandwidth**: Lower and upper bandwidth of sparse matrices
//! - **Symmetry**: Structural and numerical symmetry detection
//! - **Diagonal Dominance**: Row and column diagonal dominance
//! - **Fill-in**: Estimate fill-in for factorizations
//! - **Block Structure**: Detect block-diagonal or banded structure
//!
//! ## Examples
//!
//! ```
//! use tenrso_sparse::{CsrMatrix, patterns};
//! use tenrso_core::DenseND;
//! use scirs2_core::ndarray_ext::array;
//!
//! let data = array![[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]];
//! let dense = DenseND::from_array(data.into_dyn());
//! let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
//!
//! // Check if matrix is symmetric
//! let is_sym = patterns::is_structurally_symmetric(&mat);
//! println!("Structurally symmetric: {}", is_sym);
//!
//! // Compute bandwidth
//! let (lower, upper) = patterns::bandwidth(&mat);
//! println!("Bandwidth: lower={}, upper={}", lower, upper);
//! ```

use crate::CsrMatrix;
use scirs2_core::numeric::Float;
use std::collections::HashSet;

/// Pattern analysis statistics for a sparse matrix.
#[derive(Debug, Clone)]
pub struct PatternStats {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    /// Number of non-zeros
    pub nnz: usize,
    /// Lower bandwidth
    pub lower_bandwidth: usize,
    /// Upper bandwidth
    pub upper_bandwidth: usize,
    /// Is structurally symmetric
    pub structurally_symmetric: bool,
    /// Is numerically symmetric (if applicable)
    pub numerically_symmetric: Option<bool>,
    /// Is diagonally dominant (row-wise)
    pub row_diagonally_dominant: bool,
    /// Is diagonally dominant (column-wise)
    pub col_diagonally_dominant: bool,
    /// Average non-zeros per row
    pub avg_nnz_per_row: f64,
    /// Maximum non-zeros per row
    pub max_nnz_per_row: usize,
    /// Minimum non-zeros per row
    pub min_nnz_per_row: usize,
}

/// Computes the lower and upper bandwidth of a sparse matrix.
///
/// The bandwidth indicates how far non-zero elements are from the main diagonal.
/// - Lower bandwidth: max(i - j) for all non-zero elements (i,j)
/// - Upper bandwidth: max(j - i) for all non-zero elements (i,j)
///
/// # Complexity
///
/// O(nnz)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, patterns};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 5.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let (lower, upper) = patterns::bandwidth(&mat);
/// assert_eq!(lower, 0); // No elements below diagonal
/// assert_eq!(upper, 1); // Elements up to 1 column above diagonal
/// ```
pub fn bandwidth<T: Float>(matrix: &CsrMatrix<T>) -> (usize, usize) {
    let mut lower_bw = 0;
    let mut upper_bw = 0;

    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();

    for i in 0..matrix.nrows() {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        for &j in &col_indices[start..end] {
            if j < i {
                lower_bw = lower_bw.max(i - j);
            } else if j > i {
                upper_bw = upper_bw.max(j - i);
            }
        }
    }

    (lower_bw, upper_bw)
}

/// Checks if a sparse matrix is structurally symmetric.
///
/// A matrix is structurally symmetric if for every non-zero element (i,j),
/// there exists a non-zero element (j,i), regardless of values.
///
/// # Complexity
///
/// O(nnz) average case, O(nnz²) worst case
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, patterns};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0], [2.0, 3.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// assert!(patterns::is_structurally_symmetric(&mat));
/// ```
pub fn is_structurally_symmetric<T: Float>(matrix: &CsrMatrix<T>) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }

    let n = matrix.nrows();
    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();

    // Build a set of all (row, col) pairs
    let mut pattern = HashSet::new();
    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for &j in &col_indices[start..end] {
            pattern.insert((i, j));
        }
    }

    // Check if transpose pattern exists
    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for &j in &col_indices[start..end] {
            if !pattern.contains(&(j, i)) {
                return false;
            }
        }
    }

    true
}

/// Checks if a sparse matrix is numerically symmetric.
///
/// A matrix is numerically symmetric if `A[i,j] = A[j,i]` for all elements
/// within a given tolerance.
///
/// # Complexity
///
/// O(nnz) average case
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, patterns};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0], [2.0, 3.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// assert!(patterns::is_numerically_symmetric(&mat, 1e-10));
/// ```
pub fn is_numerically_symmetric<T: Float>(matrix: &CsrMatrix<T>, tol: T) -> bool {
    if !is_structurally_symmetric(matrix) {
        return false;
    }

    let n = matrix.nrows();
    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();
    let values = matrix.values();

    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        for idx in start..end {
            let j = col_indices[idx];
            let val_ij = values[idx];

            // Find `A[j,i]`
            let j_start = row_ptr[j];
            let j_end = row_ptr[j + 1];

            let mut found = false;
            for jdx in j_start..j_end {
                if col_indices[jdx] == i {
                    let val_ji = values[jdx];
                    if (val_ij - val_ji).abs() > tol {
                        return false;
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                return false;
            }
        }
    }

    true
}

/// Checks if a sparse matrix is row diagonally dominant.
///
/// A matrix is row diagonally dominant if for each row i:
/// `|A[i,i]| >= Σ(j≠i) |A[i,j]|`
///
/// # Complexity
///
/// O(nnz)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, patterns};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[4.0, 1.0], [1.0, 3.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// assert!(patterns::is_row_diagonally_dominant(&mat));
/// ```
pub fn is_row_diagonally_dominant<T: Float>(matrix: &CsrMatrix<T>) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }

    let n = matrix.nrows();
    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();
    let values = matrix.values();

    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        let mut diag_val = T::zero();
        let mut off_diag_sum = T::zero();

        for idx in start..end {
            let j = col_indices[idx];
            let val = values[idx];

            if i == j {
                diag_val = val.abs();
            } else {
                off_diag_sum = off_diag_sum + val.abs();
            }
        }

        if diag_val < off_diag_sum {
            return false;
        }
    }

    true
}

/// Checks if a sparse matrix is column diagonally dominant.
///
/// A matrix is column diagonally dominant if for each column j:
/// `|A[j,j]| >= Σ(i≠j) |A[i,j]|`
///
/// # Complexity
///
/// O(nnz)
pub fn is_col_diagonally_dominant<T: Float>(matrix: &CsrMatrix<T>) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }

    let n = matrix.nrows();
    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();
    let values = matrix.values();

    // Compute column sums
    let mut diag_vals = vec![T::zero(); n];
    let mut col_sums = vec![T::zero(); n];

    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];

        for idx in start..end {
            let j = col_indices[idx];
            let val = values[idx].abs();

            if i == j {
                diag_vals[j] = val;
            } else {
                col_sums[j] = col_sums[j] + val;
            }
        }
    }

    // Check diagonal dominance
    for j in 0..n {
        if diag_vals[j] < col_sums[j] {
            return false;
        }
    }

    true
}

/// Computes comprehensive pattern statistics for a sparse matrix.
///
/// # Complexity
///
/// O(nnz)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, patterns};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let stats = patterns::analyze_pattern(&mat, Some(1e-10));
/// println!("Pattern statistics: {:?}", stats);
/// ```
pub fn analyze_pattern<T: Float>(matrix: &CsrMatrix<T>, tol: Option<T>) -> PatternStats {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let nnz = matrix.nnz();

    let (lower_bandwidth, upper_bandwidth) = bandwidth(matrix);
    let structurally_symmetric = is_structurally_symmetric(matrix);
    let numerically_symmetric = tol.map(|t| is_numerically_symmetric(matrix, t));

    let row_diagonally_dominant = if nrows == ncols {
        is_row_diagonally_dominant(matrix)
    } else {
        false
    };

    let col_diagonally_dominant = if nrows == ncols {
        is_col_diagonally_dominant(matrix)
    } else {
        false
    };

    // Compute nnz per row statistics
    let row_ptr = matrix.row_ptr();
    let mut max_nnz_per_row = 0;
    let mut min_nnz_per_row = usize::MAX;

    for i in 0..nrows {
        let row_nnz = row_ptr[i + 1] - row_ptr[i];
        max_nnz_per_row = max_nnz_per_row.max(row_nnz);
        min_nnz_per_row = min_nnz_per_row.min(row_nnz);
    }

    if nrows == 0 {
        min_nnz_per_row = 0;
    }

    let avg_nnz_per_row = if nrows > 0 {
        nnz as f64 / nrows as f64
    } else {
        0.0
    };

    PatternStats {
        nrows,
        ncols,
        nnz,
        lower_bandwidth,
        upper_bandwidth,
        structurally_symmetric,
        numerically_symmetric,
        row_diagonally_dominant,
        col_diagonally_dominant,
        avg_nnz_per_row,
        max_nnz_per_row,
        min_nnz_per_row,
    }
}

/// Detects if a matrix has a banded structure.
///
/// Returns true if the matrix is banded with specified lower and upper bandwidths.
///
/// # Complexity
///
/// O(nnz)
pub fn is_banded<T: Float>(matrix: &CsrMatrix<T>, lower: usize, upper: usize) -> bool {
    let (actual_lower, actual_upper) = bandwidth(matrix);
    actual_lower <= lower && actual_upper <= upper
}

/// Estimates the fill-in for LU factorization using a simple heuristic.
///
/// This provides a rough estimate based on the sparsity pattern.
///
/// # Complexity
///
/// O(nnz)
pub fn estimate_lu_fill<T: Float>(matrix: &CsrMatrix<T>) -> usize {
    if matrix.nrows() != matrix.ncols() {
        return 0;
    }

    let (lower_bw, upper_bw) = bandwidth(matrix);
    let n = matrix.nrows();

    // Simplified estimate: bandwidth-based
    // Fill-in is roughly O(bandwidth²) per row
    let avg_fill_per_row = (lower_bw + upper_bw + 1).min(n);
    n * avg_fill_per_row
}

/// Computes the number of non-zeros per row.
///
/// # Complexity
///
/// O(nrows)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CsrMatrix, patterns};
/// use tenrso_core::DenseND;
/// use scirs2_core::ndarray_ext::array;
///
/// let data = array![[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]];
/// let dense = DenseND::from_array(data.into_dyn());
/// let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();
///
/// let nnz_per_row = patterns::nnz_per_row(&mat);
/// assert_eq!(nnz_per_row, vec![2, 1, 1]);
/// ```
pub fn nnz_per_row<T: Float>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    let row_ptr = matrix.row_ptr();
    (0..matrix.nrows())
        .map(|i| row_ptr[i + 1] - row_ptr[i])
        .collect()
}

/// Computes the number of non-zeros per column.
///
/// # Complexity
///
/// O(nnz)
pub fn nnz_per_col<T: Float>(matrix: &CsrMatrix<T>) -> Vec<usize> {
    let mut col_counts = vec![0; matrix.ncols()];
    let row_ptr = matrix.row_ptr();
    let col_indices = matrix.col_indices();

    for i in 0..matrix.nrows() {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for &j in &col_indices[start..end] {
            col_counts[j] += 1;
        }
    }

    col_counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;
    use tenrso_core::DenseND;

    #[test]
    fn test_bandwidth() {
        let data = array![[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 5.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let (lower, upper) = bandwidth(&mat);
        assert_eq!(lower, 0);
        assert_eq!(upper, 1);
    }

    #[test]
    fn test_bandwidth_tridiagonal() {
        let data = array![
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let (lower, upper) = bandwidth(&mat);
        assert_eq!(lower, 1);
        assert_eq!(upper, 1);
    }

    #[test]
    fn test_structural_symmetry() {
        let data = array![[1.0, 2.0], [2.0, 3.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(is_structurally_symmetric(&mat));
    }

    #[test]
    fn test_structural_asymmetry() {
        let data = array![[1.0, 2.0], [0.0, 3.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(!is_structurally_symmetric(&mat));
    }

    #[test]
    fn test_numerical_symmetry() {
        let data = array![[1.0, 2.0], [2.0, 3.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(is_numerically_symmetric(&mat, 1e-10));
    }

    #[test]
    fn test_numerical_asymmetry() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(!is_numerically_symmetric(&mat, 1e-10));
    }

    #[test]
    fn test_row_diagonal_dominance() {
        let data = array![[4.0, 1.0], [1.0, 3.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(is_row_diagonally_dominant(&mat));
    }

    #[test]
    fn test_row_diagonal_non_dominance() {
        let data = array![[1.0, 2.0], [2.0, 1.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(!is_row_diagonally_dominant(&mat));
    }

    #[test]
    fn test_col_diagonal_dominance() {
        let data = array![[4.0, 1.0], [1.0, 3.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(is_col_diagonally_dominant(&mat));
    }

    #[test]
    fn test_analyze_pattern() {
        let data = array![[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let stats = analyze_pattern(&mat, Some(1e-10));
        assert_eq!(stats.nrows, 3);
        assert_eq!(stats.ncols, 3);
        assert_eq!(stats.nnz, mat.nnz()); // Use actual nnz
        assert!(stats.structurally_symmetric);
        assert!(stats.numerically_symmetric.unwrap());
        assert!(stats.row_diagonally_dominant);
    }

    #[test]
    fn test_is_banded() {
        let data = array![
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        assert!(is_banded(&mat, 1, 1));
        assert!(!is_banded(&mat, 0, 1));
        assert!(!is_banded(&mat, 1, 0));
    }

    #[test]
    fn test_nnz_per_row() {
        let data = array![[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let nnz = nnz_per_row(&mat);
        assert_eq!(nnz, vec![2, 1, 1]);
    }

    #[test]
    fn test_nnz_per_col() {
        let data = array![[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]];
        let dense = DenseND::from_array(data.into_dyn());
        let mat = CsrMatrix::from_dense(&dense, 0.0).unwrap();

        let nnz = nnz_per_col(&mat);
        assert_eq!(nnz, vec![2, 2]);
    }

    #[test]
    fn test_empty_matrix() {
        let mat = CsrMatrix::<f64>::new(vec![0, 0, 0], vec![], vec![], (2, 2)).unwrap();

        assert_eq!(bandwidth(&mat), (0, 0));
        assert!(is_structurally_symmetric(&mat));
        assert!(is_numerically_symmetric(&mat, 1e-10));
    }
}
