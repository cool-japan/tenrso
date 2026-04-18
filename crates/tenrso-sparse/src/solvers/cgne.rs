//! CGNE (CG on Normal Equations) solver for least-squares problems.

use super::helpers::{axpy, dot, norm, spmv_transpose, spmv_vec};
use super::SolverInfo;
use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;

/// CGNE solver for sparse least squares problems (overdetermined systems)
///
/// Solves the least squares problem min ||Ax - b||² where A can be rectangular
/// by solving the normal equations A^T A x = A^T b using Conjugate Gradient.
///
/// This method is suitable for overdetermined systems (m > n) and is simpler
/// and more reliable than LSQR, though it squares the condition number.
///
/// # Complexity
///
/// O((nnz(A) + nnz(A^T)) × iterations) time, O(m + n) additional space
///
/// # Arguments
///
/// - `a`: Sparse matrix (m × n, typically m > n)
/// - `b`: Right-hand side vector (length m)
/// - `max_iter`: Maximum number of iterations
/// - `tol`: Convergence tolerance (relative residual norm)
///
/// # Returns
///
/// `(x, info)` where x is the least squares solution and info contains convergence information
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers};
///
/// // Overdetermined system: 4x2 matrix
/// let row_ptr = vec![0, 2, 4, 6, 8];
/// let col_indices = vec![0, 1, 0, 1, 0, 1, 0, 1];
/// let values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 2)).unwrap();
/// let b = vec![1.0, 2.0, 3.0, 1.0];
///
/// let (x, info) = solvers::cgne(&a, &b, 100, 1e-6).unwrap();
/// assert!(info.converged);
/// assert_eq!(x.len(), 2);
/// ```
pub fn cgne<T: Float>(
    a: &CsrMatrix<T>,
    b: &[T],
    max_iter: usize,
    tol: f64,
) -> SparseResult<(Vec<T>, SolverInfo)> {
    let (m, n) = a.shape();
    if b.len() != m {
        return Err(SparseError::validation(&format!(
            "RHS size {} != matrix rows {}",
            b.len(),
            m
        )));
    }

    // Solve A^T A x = A^T b using CG
    // This is equivalent to minimizing ||Ax - b||²

    // Initial guess x = 0
    let mut x = vec![T::zero(); n];

    // r = A^T b - A^T A x = A^T b (since x = 0)
    let mut r = spmv_transpose(a, b)?;

    // p = r
    let mut p = r.clone();

    // rr = <r, r>
    let mut rr = dot(&r, &r);

    let tol_sq = tol * tol;
    let b_norm = norm(b).to_f64().unwrap_or(0.0);

    for iter in 0..max_iter {
        // q = A^T A p = A^T (A p)
        let ap = spmv_vec(a, &p)?;
        let q = spmv_transpose(a, &ap)?;

        // alpha = <r, r> / <p, q>
        let pq = dot(&p, &q);
        if pq.abs() < T::epsilon() {
            return Err(SparseError::operation("CGNE: division by zero (pq)"));
        }
        let alpha = rr / pq;

        // x = x + alpha * p
        axpy(alpha, &p, &mut x);

        // r = r - alpha * q
        axpy(-alpha, &q, &mut r);

        // Check convergence: ||A^T (Ax - b)||
        let r_norm = norm(&r).to_f64().unwrap_or(f64::INFINITY);
        let relative_error = if b_norm > 0.0 {
            r_norm / b_norm
        } else {
            r_norm
        };

        if relative_error < tol || r_norm * r_norm < tol_sq {
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
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rr = rr_new;
    }

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
    use super::*;

    #[test]
    fn test_cgne_overdetermined() {
        // Overdetermined system: 4x2 matrix
        // A = [[1, 0], [0, 1], [1, 1], [1, -1]]
        let row_ptr = vec![0, 2, 4, 6, 8];
        let col_indices = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 2)).unwrap();

        let b = vec![1.0, 2.0, 3.0, 1.0];

        let (x, info) = cgne(&a, &b, 100, 1e-6).unwrap();

        assert!(info.converged);
        assert_eq!(x.len(), 2);
        assert!(info.iterations <= 10);

        // Verify that CGNE gives a least squares solution
        // The residual might not be zero for overdetermined systems
        let ax = spmv_vec(&a, &x).unwrap();
        let residual_vec: Vec<f64> = ax.iter().zip(b.iter()).map(|(axi, bi)| axi - bi).collect();
        let residual_norm: f64 = residual_vec.iter().map(|r| r * r).sum::<f64>().sqrt();

        // Should give a reasonable least squares solution (not exact for overdetermined)
        assert!(residual_norm < 3.0);
    }

    #[test]
    fn test_cgne_square() {
        // Square full-rank system: 3x3
        let row_ptr = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![3.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 3.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = vec![1.0, 2.0, 3.0];

        let (x, info) = cgne(&a, &b, 100, 1e-6).unwrap();

        assert!(info.converged);

        // For square full-rank system, CGNE should solve Ax = b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cgne_regression() {
        // Linear regression problem: fit y = mx + c
        // Points: (0,1), (1,2), (2,4), (3,5)
        // Design matrix A = [[0, 1], [1, 1], [2, 1], [3, 1]]
        let row_ptr = vec![0, 2, 4, 6, 8];
        let col_indices = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let values = vec![0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (4, 2)).unwrap();

        let b = vec![1.0, 2.0, 4.0, 5.0];

        let (x, info) = cgne(&a, &b, 100, 1e-6).unwrap();

        assert!(info.converged);

        // Expected solution: slope ≈ 1.5, intercept ≈ 1.0
        assert!((x[0] - 1.5).abs() < 0.5); // slope
        assert!((x[1] - 1.0).abs() < 0.5); // intercept
    }

    #[test]
    fn test_cgne_size_mismatch() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 2.0, 3.0]; // Wrong size

        let result = cgne(&a, &b, 100, 1e-6);
        assert!(result.is_err());
    }
}
