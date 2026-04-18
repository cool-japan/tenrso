//! BiCGSTAB solver for general nonsymmetric systems.

use super::helpers::{axpy, dot, norm_squared, spmv_vec};
use super::preconditioner::Preconditioner;
use super::SolverInfo;
use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;

/// BiCGSTAB solver for general nonsymmetric systems
///
/// Solves Ax = b where A can be nonsymmetric.
///
/// # Complexity
///
/// O(nnz × iterations) time, O(n) additional space
///
/// # Arguments
///
/// - `a`: Sparse matrix
/// - `b`: Right-hand side vector
/// - `max_iter`: Maximum number of iterations
/// - `tol`: Convergence tolerance
/// - `precond`: Optional preconditioner
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers, solvers::IdentityPreconditioner};
///
/// let row_ptr = vec![0, 2, 4];
/// let col_indices = vec![0, 1, 0, 1];
/// let values = vec![3.0, -1.0, -1.0, 2.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
/// let b = vec![1.0, 1.0];
///
/// let (x, info) = solvers::bicgstab::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();
/// assert!(info.converged);
/// ```
pub fn bicgstab<T: Float, P: Preconditioner<T>>(
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

    // r = b - A*x = b
    let mut r = b.to_vec();

    // Choose r_tilde (shadow residual) = r
    let r_tilde = r.clone();

    // rho = <r_tilde, r>
    let mut rho = dot(&r_tilde, &r);

    // p = r
    let mut p = r.clone();

    let tol_sq = tol * tol;

    for iter in 0..max_iter {
        // z = M^{-1} * p
        let z = match precond {
            Some(pc) => pc.apply(&p)?,
            None => p.clone(),
        };

        // v = A * z
        let v = spmv_vec(a, &z)?;

        // alpha = rho / <r_tilde, v>
        let rtv = dot(&r_tilde, &v);
        if rtv.abs() < T::epsilon() {
            return Err(SparseError::operation("BiCGSTAB: breakdown (rtv)"));
        }
        let alpha = rho / rtv;

        // s = r - alpha * v
        let mut s = r.clone();
        axpy(-alpha, &v, &mut s);

        // Check for early convergence
        let s_norm_sq = norm_squared(&s);
        if s_norm_sq.to_f64().unwrap_or(f64::INFINITY) < tol_sq {
            // x = x + alpha * z
            axpy(alpha, &z, &mut x);

            return Ok((
                x,
                SolverInfo {
                    iterations: iter + 1,
                    residual: s_norm_sq.to_f64().unwrap_or(0.0).sqrt(),
                    converged: true,
                },
            ));
        }

        // y = M^{-1} * s
        let y = match precond {
            Some(pc) => pc.apply(&s)?,
            None => s.clone(),
        };

        // t = A * y
        let t = spmv_vec(a, &y)?;

        // omega = <t, s> / <t, t>
        let ts = dot(&t, &s);
        let tt = dot(&t, &t);
        if tt.abs() < T::epsilon() {
            return Err(SparseError::operation("BiCGSTAB: breakdown (tt)"));
        }
        let omega = ts / tt;

        // x = x + alpha * z + omega * y
        axpy(alpha, &z, &mut x);
        axpy(omega, &y, &mut x);

        // r = s - omega * t
        r = s;
        axpy(-omega, &t, &mut r);

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

        // rho_new = <r_tilde, r>
        let rho_new = dot(&r_tilde, &r);

        // beta = (rho_new / rho) * (alpha / omega)
        if rho.abs() < T::epsilon() || omega.abs() < T::epsilon() {
            return Err(SparseError::operation("BiCGSTAB: breakdown (rho/omega)"));
        }
        let beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        rho = rho_new;
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
    use super::super::preconditioner::IdentityPreconditioner;
    use super::*;

    #[test]
    fn test_bicgstab_simple() {
        // Non-symmetric system
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![3.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];

        let (x, info) = bicgstab::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info.converged);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bicgstab_max_iterations() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![3.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
        let b = vec![1.0, 1.0];

        let (_, info) = bicgstab::<f64, IdentityPreconditioner>(&a, &b, 1, 1e-12, None).unwrap();
        assert!(!info.converged);
    }
}
