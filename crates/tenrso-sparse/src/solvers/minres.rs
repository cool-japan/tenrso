//! MINRES (Minimum Residual) solver for symmetric (possibly indefinite) systems.

use super::helpers::{axpy, dot, norm, spmv_vec};
use super::preconditioner::Preconditioner;
use super::SolverInfo;
use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;

/// MINRES solver (Minimum Residual Method)
///
/// Solves Ax = b for symmetric (possibly indefinite) matrices using the MINRES algorithm.
/// Unlike CG which requires positive definiteness, MINRES works for any symmetric matrix.
///
/// This is particularly useful for:
/// - Saddle-point problems (common in constrained optimization and fluid dynamics)
/// - Symmetric indefinite systems (where CG fails)
/// - Constrained optimization problems with Lagrange multipliers
/// - Problems arising from mixed finite element formulations
///
/// # Algorithm
///
/// MINRES uses a 3-term Lanczos recurrence to build an orthogonal basis and
/// minimizes the residual norm over the Krylov subspace. It is mathematically
/// equivalent to applying CG to the normal equations without squaring the
/// condition number.
///
/// # Complexity
///
/// O(nnz × iterations) time, O(n) space
///
/// # Arguments
///
/// - `a`: Symmetric sparse matrix (symmetry not verified, user must ensure)
/// - `b`: Right-hand side vector
/// - `max_iter`: Maximum number of iterations
/// - `tol`: Convergence tolerance (relative residual norm)
/// - `precond`: Optional symmetric preconditioner
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers, solvers::IdentityPreconditioner};
///
/// // Symmetric indefinite matrix: A = [[1, 2], [2, -1]]
/// let row_ptr = vec![0, 2, 4];
/// let col_indices = vec![0, 1, 0, 1];
/// let values = vec![1.0, 2.0, 2.0, -1.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
/// let b = vec![1.0, 1.0];
///
/// let (x, info) = solvers::minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();
/// assert!(info.converged);
/// // Verify solution: A*x ≈ b
/// ```
pub fn minres<T: Float, P: Preconditioner<T>>(
    a: &CsrMatrix<T>,
    b: &[T],
    max_iter: usize,
    tol: f64,
    precond: Option<&P>,
) -> SparseResult<(Vec<T>, SolverInfo)> {
    // Reference: C. C. Paige and M. A. Saunders, "Solution of sparse indefinite
    // systems of linear equations", SIAM J. Numer. Anal. 12(4), pp. 617-629 (1975).
    // This implementation follows the canonical MATLAB/SciPy formulation that
    // tracks the QR factorization of the Lanczos tridiagonal via scalar
    // recurrences (dbar, epsln) instead of explicit two-rotation history.
    let n = a.shape().0;
    if b.len() != n {
        return Err(SparseError::validation(&format!(
            "RHS size {} != matrix size {}",
            b.len(),
            n
        )));
    }

    // Initial guess x = 0 so that the unpreconditioned residual r = b.
    let mut x = vec![T::zero(); n];
    let b_norm = norm(b);
    if b_norm.to_f64().unwrap_or(0.0) < 1e-14 {
        return Ok((
            x,
            SolverInfo {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
        ));
    }

    // r1 = b - A*x = b; y = M^{-1} * r1 (y == r1 when no preconditioner).
    let r1_init = b.to_vec();
    let y_init = match precond {
        Some(p) => p.apply(&r1_init)?,
        None => r1_init.clone(),
    };

    // beta1 = sqrt(<r1, y>). For symmetric positive-definite M this is the
    // M^{-1}-norm of the initial residual; without preconditioning it is the
    // plain 2-norm.
    let beta1_sq = dot(&r1_init, &y_init);
    if beta1_sq.to_f64().unwrap_or(0.0) < 0.0 {
        return Err(SparseError::operation(
            "MINRES: indefinite preconditioner (⟨r, M^{-1} r⟩ < 0)",
        ));
    }
    let beta1 = beta1_sq.sqrt();
    if beta1.to_f64().unwrap_or(0.0) < 1e-14 {
        return Ok((
            x,
            SolverInfo {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
        ));
    }

    // Running state for the Lanczos recurrence (preconditioned form).
    let mut r1 = vec![T::zero(); n];
    let mut r2 = r1_init;
    let mut y = y_init;

    // Direction vectors for the three-term solution update recurrence.
    let mut w = vec![T::zero(); n];
    let mut w2 = vec![T::zero(); n];

    // Scalars tracking the rotated tridiagonal: dbar is the pre-rotation
    // diagonal carried from the previous iteration, epsln is the two-off
    // super-diagonal entry produced two iterations earlier.
    let mut beta = beta1;
    let mut oldb = T::zero();
    let mut dbar = T::zero();
    let mut epsln = T::zero();
    let mut phibar = beta1;

    // Previous Givens rotation. The initial value cs = -1 matches the
    // Paige-Saunders reference so that the first iteration's `delta` works
    // out to zero and `gbar` equals alpha_1, exactly as if no rotation had
    // been applied yet.
    let mut cs = -T::one();
    let mut sn = T::zero();

    for iter in 0..max_iter {
        // Normalise to obtain v_k = y / beta.
        let inv_beta = T::one() / beta;
        let v: Vec<T> = y.iter().map(|&yi| yi * inv_beta).collect();

        // y <- A * v_k. We keep the unpreconditioned Lanczos vector in r2
        // and the preconditioned one in y.
        let mut y_new = spmv_vec(a, &v)?;
        if iter >= 1 {
            // y <- y - (beta/oldb) * r1
            let coef = beta / oldb;
            axpy(-coef, &r1, &mut y_new);
        }

        let alfa = dot(&v, &y_new);
        // y <- y - (alfa/beta) * r2
        let coef = alfa / beta;
        axpy(-coef, &r2, &mut y_new);

        // Rotate the residual vectors: r1 becomes the previous r2, r2 becomes
        // the freshly computed unpreconditioned Lanczos vector.
        r1 = r2;
        r2 = y_new;
        y = match precond {
            Some(p) => p.apply(&r2)?,
            None => r2.clone(),
        };

        oldb = beta;
        let beta_sq = dot(&r2, &y);
        if beta_sq.to_f64().unwrap_or(0.0) < 0.0 {
            return Err(SparseError::operation(
                "MINRES: matrix is not symmetric (⟨r, M^{-1} r⟩ < 0)",
            ));
        }
        beta = beta_sq.sqrt();

        // Apply the previous Givens rotation to the new column of the
        // tridiagonal matrix T_k. See Paige & Saunders (1975), eq. (3.3).
        //   [delta_k       epsln_{k+1}] = [cs  sn] [ dbar_k       0        ]
        //   [gbar_k        dbar_{k+1} ]   [sn -cs] [ alfa_k       beta_{k+1}]
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        // Compute the new Givens rotation Q_k that eliminates beta_{k+1}.
        let gamma_unclamped = (gbar * gbar + beta * beta).sqrt();
        let gamma = if gamma_unclamped.abs() < T::epsilon() {
            T::epsilon()
        } else {
            gamma_unclamped
        };
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar = sn * phibar;

        // Update the solution using the three-term w-recurrence:
        //   w_k = (v_k - oldeps * w_{k-2} - delta * w_{k-1}) / gamma
        //   x_k = x_{k-1} + phi * w_k
        let w1 = w2;
        w2 = w;
        let mut w_new = vec![T::zero(); n];
        let inv_gamma = T::one() / gamma;
        for i in 0..n {
            w_new[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * inv_gamma;
        }
        axpy(phi, &w_new, &mut x);
        w = w_new;

        // Convergence test: ||r_k|| is tracked by |phibar| after the rotation.
        let rnorm = phibar.abs().to_f64().unwrap_or(f64::INFINITY);
        let rel = rnorm / b_norm.to_f64().unwrap_or(1.0);
        if rel < tol {
            return Ok((
                x,
                SolverInfo {
                    iterations: iter + 1,
                    residual: rnorm,
                    converged: true,
                },
            ));
        }

        // Handle Lanczos breakdown: beta_{k+1} ≈ 0 means the Krylov subspace
        // is invariant and the current x is exact (in the subspace sense).
        if beta.abs() < T::epsilon() {
            return Ok((
                x,
                SolverInfo {
                    iterations: iter + 1,
                    residual: rnorm,
                    converged: true,
                },
            ));
        }
    }

    let final_residual = phibar.abs().to_f64().unwrap_or(f64::INFINITY);
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
    use super::super::cg::cg;
    use super::super::preconditioner::{IdentityPreconditioner, JacobiPreconditioner};
    use super::*;

    #[test]
    fn test_minres_symmetric_indefinite() {
        // Symmetric indefinite matrix: A = [[1, 2], [2, -1]]
        // Eigenvalues: approximately 2.236 and -2.236 (indefinite!)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 2.0, -1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];

        let (x, info) = minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info.converged);
        assert!(info.iterations <= 10);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_minres_spd() {
        // MINRES should also work for SPD matrices
        // A = [[4, -1], [-1, 4]] (SPD)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];

        let (x, info) = minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info.converged);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_minres_tridiagonal_indefinite() {
        // Tridiagonal symmetric indefinite system
        // A = [[ 1, -2,  0],
        //      [-2,  1, -2],
        //      [ 0, -2,  1]]
        let row_ptr = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![1.0, -2.0, -2.0, 1.0, -2.0, -2.0, 1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (3, 3)).unwrap();

        let b = vec![1.0, 2.0, 3.0];

        let (x, info) = minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info.converged);
        assert!(info.iterations <= 15);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_minres_with_preconditioner() {
        // Test MINRES with Jacobi preconditioner
        // A = [[4, -1], [-1, 4]] (symmetric)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];
        let precond = JacobiPreconditioner::from_matrix(&a).unwrap();

        let (x, info) = minres(&a, &b, 100, 1e-6, Some(&precond)).unwrap();

        assert!(info.converged);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_minres_zero_rhs() {
        // Test MINRES with zero right-hand side
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 2.0, -1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![0.0, 0.0];

        let (x, info) = minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info.converged);
        assert_eq!(info.iterations, 0);

        // Solution should be zero vector
        assert!(x[0].abs() < 1e-10);
        assert!(x[1].abs() < 1e-10);
    }

    #[test]
    fn test_minres_size_mismatch() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 2.0, 2.0, -1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0, 1.0]; // Wrong size

        let result = minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_minres_vs_cg_on_spd() {
        // Compare MINRES and CG on SPD system - should give similar results
        // A = [[4, -1], [-1, 4]]
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![3.0, 1.0];

        let (x_minres, info_minres) =
            minres::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();
        let (x_cg, info_cg) = cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None).unwrap();

        assert!(info_minres.converged);
        assert!(info_cg.converged);

        // Solutions should be close
        for i in 0..2 {
            assert!((x_minres[i] - x_cg[i]).abs() < 1e-4);
        }
    }
}
