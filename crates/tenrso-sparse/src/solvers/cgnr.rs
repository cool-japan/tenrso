//! CGNR (CG on Normal Residual) solver for underdetermined least-squares problems.

use super::helpers::{axpy, dot, norm, spmv_transpose, spmv_vec};
use super::SolverInfo;
use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;

/// CGNR solver for sparse least squares problems (underdetermined systems)
///
/// Solves the least squares problem min ||Ax - b||² where A can be rectangular
/// by solving A A^T y = b, then computing x = A^T y using Conjugate Gradient.
///
/// This method is suitable for underdetermined systems (m < n) and finds the
/// minimum-norm solution. It is simpler than LSQR but squares the condition number.
///
/// # Complexity
///
/// O((nnz(A) + nnz(A^T)) × iterations) time, O(m + n) additional space
///
/// # Arguments
///
/// - `a`: Sparse matrix (m × n, typically m < n)
/// - `b`: Right-hand side vector (length m)
/// - `max_iter`: Maximum number of iterations
/// - `tol`: Convergence tolerance (relative residual norm)
///
/// # Returns
///
/// `(x, info)` where x is the minimum-norm least squares solution
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers};
///
/// // Underdetermined system: 2x3 matrix
/// let row_ptr = vec![0, 3, 6];
/// let col_indices = vec![0, 1, 2, 0, 1, 2];
/// let values = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();
/// let b = vec![2.0, 3.0];
///
/// let (x, info) = solvers::cgnr(&a, &b, 100, 1e-6).unwrap();
/// assert!(info.converged);
/// assert_eq!(x.len(), 3);
/// ```
pub fn cgnr<T: Float>(
    a: &CsrMatrix<T>,
    b: &[T],
    max_iter: usize,
    tol: f64,
) -> SparseResult<(Vec<T>, SolverInfo)> {
    let (m, _n) = a.shape();
    if b.len() != m {
        return Err(SparseError::validation(&format!(
            "RHS size {} != matrix rows {}",
            b.len(),
            m
        )));
    }

    // Solve A A^T y = b, then x = A^T y
    // This gives the minimum-norm solution for underdetermined systems

    // Initial guess y = 0
    let mut y = vec![T::zero(); m];

    // r = b - A A^T y = b (since y = 0)
    let mut r = b.to_vec();

    // p = r
    let mut p = r.clone();

    // rr = <r, r>
    let mut rr = dot(&r, &r);

    let tol_sq = tol * tol;
    let b_norm = norm(b).to_f64().unwrap_or(0.0);

    for iter in 0..max_iter {
        // q = A A^T p = A (A^T p)
        let atp = spmv_transpose(a, &p)?;
        let q = spmv_vec(a, &atp)?;

        // alpha = <r, r> / <p, q>
        let pq = dot(&p, &q);
        if pq.abs() < T::epsilon() {
            return Err(SparseError::operation("CGNR: division by zero (pq)"));
        }
        let alpha = rr / pq;

        // y = y + alpha * p
        axpy(alpha, &p, &mut y);

        // r = r - alpha * q
        axpy(-alpha, &q, &mut r);

        // Check convergence: ||Ax - b|| = ||r||
        let r_norm = norm(&r).to_f64().unwrap_or(f64::INFINITY);
        let relative_error = if b_norm > 0.0 {
            r_norm / b_norm
        } else {
            r_norm
        };

        if relative_error < tol || r_norm * r_norm < tol_sq {
            // Compute x = A^T y
            let x = spmv_transpose(a, &y)?;
            return Ok((
                x,
                SolverInfo {
                    iterations: iter + 1,
                    residual: r_norm,
                    converged: true,
                },
            ));
        }

        // rr_new = <r, r>
        let rr_new = dot(&r, &r);

        // beta = rr_new / rr
        let beta = rr_new / rr;

        // p = r + beta * p
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            p[i] = r[i] + beta * p[i];
        }

        rr = rr_new;
    }

    // Final solution x = A^T y
    let x = spmv_transpose(a, &y)?;
    let final_residual = norm(&r).to_f64().unwrap_or(f64::INFINITY);
    Ok((
        x,
        SolverInfo {
            iterations: max_iter,
            residual: final_residual,
            converged: false,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::cgne::cgne;
    use super::*;

    #[test]
    fn test_cgnr_underdetermined() {
        // Underdetermined system: 2x3 matrix
        // A = [[1, 0, 1], [0, 1, 1]]
        let row_ptr = vec![0, 3, 6];
        let col_indices = vec![0, 1, 2, 0, 1, 2];
        let values = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 3)).unwrap();

        let b = vec![2.0, 3.0];

        let (x, info) = cgnr(&a, &b, 100, 1e-6).unwrap();

        assert!(info.converged);
        assert_eq!(x.len(), 3);
        assert!(info.iterations <= 10);

        // Verify Ax = b (exact for consistent system)
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cgnr_square() {
        // Square full-rank system: 3x3
        let row_ptr = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![3.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 3.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = vec![1.0, 2.0, 3.0];

        let (x, info) = cgnr(&a, &b, 100, 1e-6).unwrap();

        assert!(info.converged);

        // For square full-rank system, CGNR should solve Ax = b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cgnr_minimum_norm() {
        // Underdetermined 1x2 system: should find minimum-norm solution
        // A = [[1, 1]]
        let row_ptr = vec![0, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0, 1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (1, 2)).unwrap();

        let b = vec![2.0];

        let (x, info) = cgnr(&a, &b, 100, 1e-6).unwrap();

        assert!(info.converged);

        // Verify Ax = b
        let ax = spmv_vec(&a, &x).unwrap();
        assert!((ax[0] - b[0]).abs() < 1e-4);

        // Minimum-norm solution for x + y = 2 is x = y = 1
        assert!((x[0] - 1.0).abs() < 1e-4);
        assert!((x[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cgnr_size_mismatch() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 2.0, 3.0]; // Wrong size

        let result = cgnr(&a, &b, 100, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_cgne_vs_cgnr_square() {
        // For square full-rank systems, CGNE and CGNR should give same solution
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![3.0, 1.0, 1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![7.0, 5.0];

        let (x_cgne, info_cgne) = cgne(&a, &b, 100, 1e-6).unwrap();
        let (x_cgnr, info_cgnr) = cgnr(&a, &b, 100, 1e-6).unwrap();

        assert!(info_cgne.converged);
        assert!(info_cgnr.converged);

        // Solutions should be close
        for i in 0..2 {
            assert!((x_cgne[i] - x_cgnr[i]).abs() < 1e-3);
        }
    }
}
