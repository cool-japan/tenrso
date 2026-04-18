//! Conjugate Gradient (CG) solver for symmetric positive-definite systems.

use super::helpers::{axpy, dot, norm_squared, spmv_vec};
use super::preconditioner::Preconditioner;
use super::SolverInfo;
use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;

/// Conjugate Gradient solver for SPD systems
///
/// Solves Ax = b where A is symmetric positive definite.
///
/// # Complexity
///
/// O(nnz × iterations) time, O(n) additional space
///
/// # Arguments
///
/// - `a`: Sparse matrix (must be SPD)
/// - `b`: Right-hand side vector
/// - `max_iter`: Maximum number of iterations
/// - `tol`: Convergence tolerance (residual norm)
/// - `precond`: Optional preconditioner
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers, solvers::IdentityPreconditioner};
///
/// let row_ptr = vec![0, 2, 4];
/// let col_indices = vec![0, 1, 0, 1];
/// let values = vec![2.0, -1.0, -1.0, 2.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
/// let b = vec![1.0, 1.0];
///
/// let (x, info) = solvers::cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();
/// assert!(info.converged);
/// ```
pub fn cg<T: Float, P: Preconditioner<T>>(
    a: &CsrMatrix<T>,
    b: &[T],
    max_iter: usize,
    tol: f64,
    precond: Option<&P>,
) -> SparseResult<(Vec<T>, SolverInfo)> {
    let n = a.shape().0;
    if b.len() != n {
        return Err(SparseError::validation(&format!(
            "RHS size {} != matrix size {}",
            b.len(),
            n
        )));
    }

    // Initial guess x = 0
    let mut x = vec![T::zero(); n];

    // r = b - A*x = b (since x = 0)
    let mut r = b.to_vec();

    // z = M^{-1} * r
    let mut z = match precond {
        Some(p) => p.apply(&r)?,
        None => r.clone(),
    };

    // p = z
    let mut p = z.clone();

    // rz = <r, z>
    let mut rz = dot(&r, &z);

    let tol_sq = tol * tol;

    for iter in 0..max_iter {
        // q = A * p
        let q = spmv_vec(a, &p)?;

        // alpha = <r, z> / <p, q>
        let pq = dot(&p, &q);
        if pq.abs() < T::epsilon() {
            return Err(SparseError::operation("CG: division by zero (pq)"));
        }
        let alpha = rz / pq;

        // x = x + alpha * p
        axpy(alpha, &p, &mut x);

        // r = r - alpha * q
        axpy(-alpha, &q, &mut r);

        // Check convergence
        let r_norm_sq = norm_squared(&r);
        if r_norm_sq.to_f64().unwrap_or(f64::INFINITY) < tol_sq {
            return Ok((
                x,
                SolverInfo {
                    iterations: iter + 1,
                    residual: r_norm_sq.to_f64().unwrap_or(0.0).sqrt(),
                    converged: true,
                },
            ));
        }

        // z = M^{-1} * r
        z = match precond {
            Some(p) => p.apply(&r)?,
            None => r.clone(),
        };

        // rz_new = <r, z>
        let rz_new = dot(&r, &z);

        // beta = rz_new / rz
        let beta = rz_new / rz;

        // p = z + beta * p
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    let r_norm = norm_squared(&r).to_f64().unwrap_or(0.0).sqrt();
    Ok((
        x,
        SolverInfo {
            iterations: max_iter,
            residual: r_norm,
            converged: false,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::preconditioner::{
        IdentityPreconditioner, IluPreconditioner, JacobiPreconditioner, SsorPreconditioner,
    };
    use super::*;

    #[test]
    fn test_cg_simple_spd() {
        // A = [[2, -1], [-1, 2]] (SPD)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![2.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];

        let (x, info) = cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info.converged);
        assert!(info.iterations <= 10);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cg_with_ilu_preconditioner() {
        // Larger SPD system
        let row_ptr = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = vec![1.0, 2.0, 3.0];

        // Create ILU preconditioner
        let precond = IluPreconditioner::from_matrix(&a).unwrap();

        let (x, info) = cg(&a, &b, 100, 1e-6, Some(&precond)).unwrap();

        assert!(info.converged);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cg_max_iterations() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![2.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
        let b = vec![1.0, 1.0];

        // Test that solver respects max_iter limit
        let (_, info) = cg::<f64, IdentityPreconditioner>(&a, &b, 1, 1e-100, None).unwrap();
        // For this simple 2x2 system, CG might converge in 1 iteration
        // Just verify that iterations <= max_iter
        assert!(info.iterations <= 1);
    }

    #[test]
    fn test_cg_size_mismatch() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![2.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
        let b = vec![1.0, 1.0, 1.0]; // Wrong size

        let result = cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_jacobi_with_cg() {
        // A = [[4, -1], [-1, 4]] (SPD)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];
        let precond = JacobiPreconditioner::from_matrix(&a).unwrap();

        let (x, info) = cg(&a, &b, 100, 1e-6, Some(&precond)).unwrap();

        assert!(info.converged);
        // Jacobi should improve convergence
        assert!(info.iterations <= 10);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ssor_with_cg() {
        // A = [[4, -1], [-1, 4]] (SPD)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];
        let precond = SsorPreconditioner::from_matrix(&a, 1.0).unwrap();

        let (x, info) = cg(&a, &b, 100, 1e-6, Some(&precond)).unwrap();

        assert!(info.converged);
        // SSOR should improve convergence
        assert!(info.iterations <= 10);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_preconditioner_comparison() {
        // Larger SPD system: A = tridiag(-1, 4, -1)
        let n = 10;
        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(4.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptr.push(col_indices.len());
        }

        let a = CsrMatrix::new(row_ptr, col_indices, values, (n, n)).unwrap();
        let b = vec![1.0; n];

        // No preconditioning
        let (_, info_none) = cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        // Jacobi preconditioning
        let precond_jacobi = JacobiPreconditioner::from_matrix(&a).unwrap();
        let (_, info_jacobi) = cg(&a, &b, 100, 1e-6, Some(&precond_jacobi)).unwrap();

        // SSOR preconditioning
        let precond_ssor = SsorPreconditioner::from_matrix(&a, 1.0).unwrap();
        let (_, info_ssor) = cg(&a, &b, 100, 1e-6, Some(&precond_ssor)).unwrap();

        // All should converge
        assert!(info_none.converged);
        assert!(info_jacobi.converged);
        assert!(info_ssor.converged);

        // Preconditioners should reduce iterations
        assert!(info_jacobi.iterations <= info_none.iterations);
        assert!(info_ssor.iterations <= info_none.iterations);
    }
}
