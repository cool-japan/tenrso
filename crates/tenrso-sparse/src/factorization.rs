//! Sparse matrix factorizations
//!
//! This module provides factorization algorithms for sparse matrices:
//! - Incomplete LU (ILU) factorization
//! - Incomplete Cholesky (IC) factorization
//!
//! These are primarily used as preconditioners for iterative solvers.
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CsrMatrix, factorization};
//! use scirs2_core::ndarray_ext::array;
//!
//! // Create a sparse matrix
//! let row_ptr = vec![0, 2, 4, 6];
//! let col_indices = vec![0, 1, 0, 1, 1, 2];
//! let values = vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0];
//! let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
//!
//! // Compute ILU(0) factorization
//! let (l, u) = factorization::ilu0(&csr).unwrap();
//! // L has lower triangular, U has diagonal + upper triangular
//! assert!(l.nnz() + u.nnz() >= csr.nnz());
//! ```

use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Incomplete LU factorization with zero fill-in (ILU(0))
///
/// Computes an incomplete LU factorization of a sparse matrix,
/// preserving only the nonzero pattern of the original matrix.
///
/// Returns `(L, U)` where:
/// - `L` is lower triangular with unit diagonal (diagonal not stored)
/// - `U` is upper triangular with explicit diagonal
///
/// # Complexity
///
/// O(nnz²) in the worst case, typically O(nnz × bandwidth)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, factorization::ilu0};
///
/// let row_ptr = vec![0, 2, 4, 6];
/// let col_indices = vec![0, 1, 0, 1, 1, 2];
/// let values = vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let (l, u) = ilu0(&csr).unwrap();
/// // L has unit diagonal, U has explicit diagonal
/// ```
pub fn ilu0<T: Float>(a: &CsrMatrix<T>) -> SparseResult<(CsrMatrix<T>, CsrMatrix<T>)> {
    let (nrows, ncols) = a.shape();
    if nrows != ncols {
        return Err(SparseError::validation("Matrix must be square for ILU"));
    }
    let n = nrows;

    // Initialize L and U with same sparsity pattern as A
    let mut l_row_ptr = vec![0];
    let mut l_col_indices = Vec::new();
    let mut l_values = Vec::new();

    let mut u_row_ptr = vec![0];
    let mut u_col_indices = Vec::new();
    let mut u_values = Vec::new();

    // Process each row
    for i in 0..n {
        let row_start = a.row_ptr()[i];
        let row_end = a.row_ptr()[i + 1];

        // Separate lower and upper parts
        let mut l_row = Vec::new();
        let mut u_row = Vec::new();

        for idx in row_start..row_end {
            let j = a.col_indices()[idx];
            let val = a.values()[idx];

            if j < i {
                // Lower part (without diagonal)
                l_row.push((j, val));
            } else {
                // Upper part (with diagonal)
                u_row.push((j, val));
            }
        }

        // Apply ILU updates using previously computed rows
        for &(k, l_ik) in &l_row {
            // Find u_kj for all j in row i where j >= k
            let u_k_start = u_row_ptr[k];
            let u_k_end = u_row_ptr[k + 1];

            // Get diagonal of U[k,k]
            let mut u_kk = T::zero();
            for idx in u_k_start..u_k_end {
                if u_col_indices[idx] == k {
                    u_kk = u_values[idx];
                    break;
                }
            }

            if u_kk.abs() < T::from(1e-14).unwrap() {
                return Err(SparseError::validation("Near-zero pivot in ILU"));
            }

            // Update u_row entries where nonzero pattern matches
            for idx in u_k_start..u_k_end {
                let j = u_col_indices[idx];
                if j <= i {
                    continue;
                }
                let u_kj = u_values[idx];

                // Update U[i,j] if it exists in the pattern
                if let Some(pos) = u_row.iter().position(|(col, _)| *col == j) {
                    u_row[pos].1 = u_row[pos].1 - l_ik * u_kj / u_kk;
                }
            }
        }

        // Normalize L row by diagonal of U
        if let Some(diag_pos) = u_row.iter().position(|(col, _)| *col == i) {
            let u_ii = u_row[diag_pos].1;
            if u_ii.abs() < T::from(1e-14).unwrap() {
                return Err(SparseError::validation("Near-zero pivot in ILU"));
            }

            for entry in &mut l_row {
                entry.1 = entry.1 / u_ii;
            }
        }

        // Store L row
        for (col, val) in l_row {
            l_col_indices.push(col);
            l_values.push(val);
        }
        l_row_ptr.push(l_col_indices.len());

        // Store U row
        for (col, val) in u_row {
            u_col_indices.push(col);
            u_values.push(val);
        }
        u_row_ptr.push(u_col_indices.len());
    }

    let l = CsrMatrix::new(l_row_ptr, l_col_indices, l_values, (n, n))?;
    let u = CsrMatrix::new(u_row_ptr, u_col_indices, u_values, (n, n))?;

    Ok((l, u))
}

/// Incomplete Cholesky factorization with zero fill-in (IC(0))
///
/// Computes an incomplete Cholesky factorization of a symmetric positive
/// definite sparse matrix, preserving only the lower triangular nonzero pattern.
///
/// Returns `L` where `A ≈ L * L^T`
///
/// # Requirements
///
/// - Matrix must be square
/// - Matrix should be symmetric positive definite
///
/// # Complexity
///
/// O(nnz²) in the worst case, typically O(nnz × bandwidth)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, factorization::ic0};
///
/// // Symmetric positive definite matrix
/// let row_ptr = vec![0, 2, 4, 6];
/// let col_indices = vec![0, 1, 0, 1, 1, 2];
/// let values = vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0];
/// let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let l = ic0(&csr).unwrap();
/// // L is lower triangular with explicit diagonal
/// ```
pub fn ic0<T: Float>(a: &CsrMatrix<T>) -> SparseResult<CsrMatrix<T>> {
    let (nrows, ncols) = a.shape();
    if nrows != ncols {
        return Err(SparseError::validation(
            "Matrix must be square for Cholesky",
        ));
    }
    let n = nrows;

    let mut l_row_ptr = vec![0];
    let mut l_col_indices = Vec::new();
    let mut l_values = Vec::new();

    // Store computed L values by (row, col) for lookup
    let mut l_map: HashMap<(usize, usize), T> = HashMap::new();

    for i in 0..n {
        let row_start = a.row_ptr()[i];
        let row_end = a.row_ptr()[i + 1];

        let mut l_row = Vec::new();

        // Collect lower triangular entries (j <= i)
        for idx in row_start..row_end {
            let j = a.col_indices()[idx];
            if j <= i {
                l_row.push((j, a.values()[idx]));
            }
        }

        // Apply IC updates
        for k in 0..l_row.len() {
            let (j, mut a_ij) = l_row[k];

            // Compute L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
            for (col_m, l_im) in l_row.iter().take(k) {
                if let Some(&l_jm) = l_map.get(&(j, *col_m)) {
                    a_ij = a_ij - *l_im * l_jm;
                }
            }

            if j == i {
                // Diagonal element
                if a_ij <= T::zero() {
                    return Err(SparseError::validation(
                        "Matrix is not positive definite for IC",
                    ));
                }
                l_row[k].1 = a_ij.sqrt();
            } else {
                // Off-diagonal element
                let l_jj = l_map
                    .get(&(j, j))
                    .ok_or_else(|| SparseError::validation("Missing diagonal in IC"))?;

                if l_jj.abs() < T::from(1e-14).unwrap() {
                    return Err(SparseError::validation("Near-zero diagonal in IC"));
                }

                l_row[k].1 = a_ij / *l_jj;
            }

            // Store in map for future lookups
            l_map.insert((i, j), l_row[k].1);
        }

        // Store L row
        for (col, val) in l_row {
            l_col_indices.push(col);
            l_values.push(val);
        }
        l_row_ptr.push(l_col_indices.len());
    }

    CsrMatrix::new(l_row_ptr, l_col_indices, l_values, (n, n)).map_err(|e| e.into())
}

/// Forward substitution for lower triangular system: L * y = b
///
/// Solves the system where `L` is lower triangular (with unit or explicit diagonal).
///
/// # Arguments
///
/// - `l`: Lower triangular matrix in CSR format
/// - `b`: Right-hand side vector
/// - `unit_diagonal`: If true, L has unit diagonal (not stored)
///
/// # Complexity
///
/// O(nnz)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, factorization::forward_substitution};
/// use scirs2_core::ndarray_ext::array;
///
/// let row_ptr = vec![0, 1, 3, 4];
/// let col_indices = vec![0, 0, 1, 2];
/// let values = vec![2.0, 1.0, 3.0, 4.0];
/// let l = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let b = array![2.0, 6.0, 8.0];
/// let y = forward_substitution(&l, &b, false).unwrap();
/// ```
pub fn forward_substitution<T: Float>(
    l: &CsrMatrix<T>,
    b: &scirs2_core::ndarray_ext::Array1<T>,
    unit_diagonal: bool,
) -> SparseResult<scirs2_core::ndarray_ext::Array1<T>> {
    let (n, _) = l.shape();
    if b.len() != n {
        return Err(SparseError::validation("Dimension mismatch in solve"));
    }

    use scirs2_core::ndarray_ext::Array1;
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let mut sum = b[i];

        let row_start = l.row_ptr()[i];
        let row_end = l.row_ptr()[i + 1];

        for idx in row_start..row_end {
            let j = l.col_indices()[idx];
            if j >= i {
                break;
            }
            sum = sum - l.values()[idx] * y[j];
        }

        if unit_diagonal {
            y[i] = sum;
        } else {
            // Find diagonal element
            let mut diag = T::zero();
            for idx in row_start..row_end {
                if l.col_indices()[idx] == i {
                    diag = l.values()[idx];
                    break;
                }
            }
            if diag.abs() < T::from(1e-14).unwrap() {
                return Err(SparseError::validation("Near-zero diagonal in solve"));
            }
            y[i] = sum / diag;
        }
    }

    Ok(y)
}

/// Backward substitution for upper triangular system: U * x = y
///
/// Solves the system where `U` is upper triangular with explicit diagonal.
///
/// # Complexity
///
/// O(nnz)
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, factorization::backward_substitution};
/// use scirs2_core::ndarray_ext::array;
///
/// let row_ptr = vec![0, 2, 3, 4];
/// let col_indices = vec![0, 2, 1, 2];
/// let values = vec![2.0, 1.0, 3.0, 4.0];
/// let u = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();
///
/// let y = array![4.0, 3.0, 8.0];
/// let x = backward_substitution(&u, &y).unwrap();
/// ```
pub fn backward_substitution<T: Float>(
    u: &CsrMatrix<T>,
    y: &scirs2_core::ndarray_ext::Array1<T>,
) -> SparseResult<scirs2_core::ndarray_ext::Array1<T>> {
    let (n, _) = u.shape();
    if y.len() != n {
        return Err(SparseError::validation("Dimension mismatch in solve"));
    }

    use scirs2_core::ndarray_ext::Array1;
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = y[i];

        let row_start = u.row_ptr()[i];
        let row_end = u.row_ptr()[i + 1];

        let mut diag = T::zero();
        for idx in row_start..row_end {
            let j = u.col_indices()[idx];
            if j == i {
                diag = u.values()[idx];
            } else if j > i {
                sum = sum - u.values()[idx] * x[j];
            }
        }

        if diag.abs() < T::from(1e-14).unwrap() {
            return Err(SparseError::validation("Near-zero diagonal in solve"));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_ilu0_simple() {
        // 3x3 matrix
        let row_ptr = vec![0, 2, 4, 6];
        let col_indices = vec![0, 1, 0, 1, 1, 2];
        let values = vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let (l, u) = ilu0(&csr).unwrap();

        // Check dimensions
        assert_eq!(l.shape(), (3, 3));
        assert_eq!(u.shape(), (3, 3));

        // L should be lower triangular
        for i in 0..3 {
            for idx in l.row_ptr()[i]..l.row_ptr()[i + 1] {
                assert!(l.col_indices()[idx] < i);
            }
        }

        // U should have diagonal
        for i in 0..3 {
            let row_start = u.row_ptr()[i];
            let row_end = u.row_ptr()[i + 1];
            let has_diag = (row_start..row_end).any(|idx| u.col_indices()[idx] == i);
            assert!(has_diag, "Row {} missing diagonal", i);
        }
    }

    #[test]
    fn test_ic0_simple() {
        // SPD matrix: [4, 1, 0; 1, 3, 1; 0, 1, 2]
        let row_ptr = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let l = ic0(&csr).unwrap();

        // Check dimensions
        assert_eq!(l.shape(), (3, 3));

        // L should be lower triangular with diagonal
        for i in 0..3 {
            let row_start = l.row_ptr()[i];
            let row_end = l.row_ptr()[i + 1];

            // All entries should be <= i
            for idx in row_start..row_end {
                assert!(l.col_indices()[idx] <= i);
            }

            // Should have diagonal
            let has_diag = (row_start..row_end).any(|idx| l.col_indices()[idx] == i);
            assert!(has_diag, "Row {} missing diagonal", i);
        }
    }

    #[test]
    fn test_forward_substitution() {
        // L = [2, 0, 0; 1, 3, 0; 0, 0, 4]
        let row_ptr = vec![0, 1, 3, 4];
        let col_indices = vec![0, 0, 1, 2];
        let values = vec![2.0, 1.0, 3.0, 4.0];
        let l = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = array![2.0, 6.0, 8.0];
        let y = forward_substitution(&l, &b, false).unwrap();

        // Verify L * y = b
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 5.0 / 3.0).abs() < 1e-10);
        assert!((y[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_substitution() {
        // U = [2, 1, 0; 0, 3, 1; 0, 0, 4]
        let row_ptr = vec![0, 2, 4, 5];
        let col_indices = vec![0, 1, 1, 2, 2];
        let values = vec![2.0, 1.0, 3.0, 1.0, 4.0];
        let u = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let y = array![3.0, 9.0, 8.0];
        let x = backward_substitution(&u, &y).unwrap();

        // Verify U * x = y
        assert!((x[2] - 2.0).abs() < 1e-10);
        assert!((x[1] - (9.0 - 2.0) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ilu0_singular_error() {
        // Singular matrix
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 0.0, 1.0]; // Zero diagonal
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        assert!(ilu0(&csr).is_err());
    }

    #[test]
    fn test_ic0_not_spd_error() {
        // Not positive definite (negative diagonal)
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![-1.0, -1.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        assert!(ic0(&csr).is_err());
    }

    #[test]
    fn test_ilu0_nonsquare_error() {
        let row_ptr = vec![0, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];
        let csr = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        assert!(ilu0(&csr).is_err());
    }
}
