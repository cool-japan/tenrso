//! GMRES (Generalized Minimal Residual) solver with restart.

use super::helpers::{axpy, dot, norm, norm_squared, spmv_vec};
use super::preconditioner::Preconditioner;
use super::SolverInfo;
use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::numeric::Float;

/// GMRES solver with restart
///
/// Solves Ax = b using the Generalized Minimal Residual method.
///
/// # Complexity
///
/// O(nnz × iterations × restart) time, O(n × restart) space
///
/// # Arguments
///
/// - `a`: Sparse matrix
/// - `b`: Right-hand side vector
/// - `max_iter`: Maximum number of restart cycles
/// - `restart`: Number of iterations before restart
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
/// let (x, info) = solvers::gmres::<f64, IdentityPreconditioner>(&a, &b, 100, 20, 1e-6, None).unwrap();
/// assert!(info.converged);
/// ```
pub fn gmres<T: Float, P: Preconditioner<T>>(
    a: &CsrMatrix<T>,
    b: &[T],
    max_iter: usize,
    restart: usize,
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

    let tol_sq = tol * tol;
    let b_norm = norm_squared(b).to_f64().unwrap_or(0.0).sqrt();

    for cycle in 0..max_iter {
        // z = M^{-1} * r
        let z = match precond {
            Some(p) => p.apply(&r)?,
            None => r.clone(),
        };

        let beta = norm(&z);
        if beta.abs() < T::epsilon() {
            return Ok((
                x,
                SolverInfo {
                    iterations: cycle * restart,
                    residual: 0.0,
                    converged: true,
                },
            ));
        }

        // V[:, 0] = z / beta
        let mut v = vec![vec![T::zero(); n]; restart + 1];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            v[0][i] = z[i] / beta;
        }

        // Hessenberg matrix
        let mut h = vec![vec![T::zero(); restart]; restart + 1];

        // Givens rotations
        let mut cs = vec![T::zero(); restart];
        let mut sn = vec![T::zero(); restart];

        // RHS of least squares problem
        let mut s = vec![T::zero(); restart + 1];
        s[0] = beta;

        // Arnoldi iteration
        let mut j = 0;
        while j < restart {
            // w = M^{-1} * A * V[:, j]
            let av = spmv_vec(a, &v[j])?;
            let w = match precond {
                Some(p) => p.apply(&av)?,
                None => av,
            };

            // Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
            }

            let mut w_orth = w.clone();
            for i in 0..=j {
                axpy(-h[i][j], &v[i], &mut w_orth);
            }

            h[j + 1][j] = norm(&w_orth);

            if h[j + 1][j].abs() > T::epsilon() {
                #[allow(clippy::needless_range_loop)]
                for i in 0..n {
                    v[j + 1][i] = w_orth[i] / h[j + 1][j];
                }
            }

            // Apply previous Givens rotations to new column of H
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }

            // Compute new Givens rotation
            let (c, sn_val) = givens(h[j][j], h[j + 1][j]);
            cs[j] = c;
            sn[j] = sn_val;

            // Apply new rotation
            h[j][j] = c * h[j][j] + sn_val * h[j + 1][j];
            h[j + 1][j] = T::zero();

            // Apply rotation to RHS vector
            let temp = c * s[j];
            s[j + 1] = -sn_val * s[j];
            s[j] = temp;

            // Check convergence
            let residual = s[j + 1].abs().to_f64().unwrap_or(f64::INFINITY);
            if residual < tol * b_norm {
                // Solve upper triangular system
                let y = backward_solve_triangular(&h, &s, j + 1)?;

                // x = x + V * y
                for i in 0..=j {
                    axpy(y[i], &v[i], &mut x);
                }

                return Ok((
                    x,
                    SolverInfo {
                        iterations: cycle * restart + j + 1,
                        residual,
                        converged: true,
                    },
                ));
            }

            j += 1;
        }

        // Solve upper triangular system
        let y = backward_solve_triangular(&h, &s, restart)?;

        // x = x + V * y
        for i in 0..restart {
            axpy(y[i], &v[i], &mut x);
        }

        // Compute new residual
        r = b.to_vec();
        let ax = spmv_vec(a, &x)?;
        axpy(-T::one(), &ax, &mut r);

        let r_norm_sq = norm_squared(&r);
        if r_norm_sq.to_f64().unwrap_or(f64::INFINITY) < tol_sq {
            return Ok((
                x,
                SolverInfo {
                    iterations: (cycle + 1) * restart,
                    residual: r_norm_sq.to_f64().unwrap_or(0.0).sqrt(),
                    converged: true,
                },
            ));
        }
    }

    let r_norm = norm_squared(&r).to_f64().unwrap_or(0.0).sqrt();
    Ok((
        x,
        SolverInfo {
            iterations: max_iter * restart,
            residual: r_norm,
            converged: false,
        },
    ))
}

/// Givens rotation parameters
fn givens<T: Float>(a: T, b: T) -> (T, T) {
    if b.abs() < T::epsilon() {
        (T::one(), T::zero())
    } else {
        let r = (a * a + b * b).sqrt();
        (a / r, b / r)
    }
}

/// Backward solve for upper triangular system H*y = s
fn backward_solve_triangular<T: Float>(h: &[Vec<T>], s: &[T], n: usize) -> SparseResult<Vec<T>> {
    let mut y = vec![T::zero(); n];

    for i in (0..n).rev() {
        let mut sum = s[i];
        #[allow(clippy::needless_range_loop)]
        for j in (i + 1)..n {
            sum = sum - h[i][j] * y[j];
        }

        if h[i][i].abs() < T::epsilon() {
            return Err(SparseError::operation("Singular triangular matrix"));
        }

        y[i] = sum / h[i][i];
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::super::preconditioner::IdentityPreconditioner;
    use super::*;

    #[test]
    fn test_gmres_simple() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![3.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let b = vec![1.0, 1.0];

        let (x, info) = gmres::<f64, IdentityPreconditioner>(&a, &b, 100, 10, 1e-6, None).unwrap();

        assert!(info.converged);

        // Check A*x ≈ b
        let ax = spmv_vec(&a, &x).unwrap();
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gmres_max_iterations() {
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![3.0, -1.0, -1.0, 2.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
        let b = vec![1.0, 1.0];

        // Extremely tight tolerance to force non-convergence
        let (_, info) = gmres::<f64, IdentityPreconditioner>(&a, &b, 1, 5, 1e-100, None).unwrap();
        assert!(!info.converged);
    }

    #[test]
    fn test_givens_rotation() {
        let (c, s) = givens(3.0, 4.0);
        assert!((c - 0.6).abs() < 1e-10);
        assert!((s - 0.8).abs() < 1e-10);

        let (c, s) = givens(1.0, 0.0);
        assert_eq!(c, 1.0);
        assert_eq!(s, 0.0);
    }
}
