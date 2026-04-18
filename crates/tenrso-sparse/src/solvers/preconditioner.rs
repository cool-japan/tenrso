//! Preconditioners for iterative sparse solvers
//!
//! Provides the [`Preconditioner`] trait together with several concrete
//! implementations: identity (no preconditioning), ILU(0), Jacobi (diagonal),
//! and SSOR (Symmetric Successive Over-Relaxation).

use crate::{CsrMatrix, SparseError, SparseResult};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::numeric::Float;

/// Preconditioner trait for iterative solvers
pub trait Preconditioner<T: Float> {
    /// Apply preconditioner: solve M*z = r for z
    fn apply(&self, r: &[T]) -> SparseResult<Vec<T>>;
}

/// Identity preconditioner (no preconditioning)
pub struct IdentityPreconditioner;

impl<T: Float> Preconditioner<T> for IdentityPreconditioner {
    fn apply(&self, r: &[T]) -> SparseResult<Vec<T>> {
        Ok(r.to_vec())
    }
}

/// ILU(0) preconditioner
pub struct IluPreconditioner<T: Float> {
    l: CsrMatrix<T>,
    u: CsrMatrix<T>,
}

impl<T: Float> IluPreconditioner<T> {
    /// Create ILU preconditioner from L and U factors
    pub fn new(l: CsrMatrix<T>, u: CsrMatrix<T>) -> Self {
        Self { l, u }
    }

    /// Create ILU preconditioner by factorizing A
    pub fn from_matrix(a: &CsrMatrix<T>) -> SparseResult<Self> {
        let (l, u) = crate::factorization::ilu0(a)?;
        Ok(Self { l, u })
    }
}

impl<T: Float> Preconditioner<T> for IluPreconditioner<T> {
    fn apply(&self, r: &[T]) -> SparseResult<Vec<T>> {
        // Solve L*U*z = r
        // First solve L*y = r (L has unit diagonal for ILU)
        let r_array = Array1::from(r.to_vec());
        let y = crate::factorization::forward_substitution(&self.l, &r_array, true)?;
        // Then solve U*z = y
        let z = crate::factorization::backward_substitution(&self.u, &y)?;
        Ok(z.to_vec())
    }
}

/// Jacobi (diagonal) preconditioner
///
/// Uses M = diag(A) as the preconditioner. This is the simplest and cheapest
/// preconditioner, but may not be effective for all problems.
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers::{JacobiPreconditioner, Preconditioner}};
///
/// let row_ptr = vec![0, 2, 4];
/// let col_indices = vec![0, 1, 0, 1];
/// let values = vec![4.0, -1.0, -1.0, 4.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
///
/// let precond = JacobiPreconditioner::from_matrix(&a).unwrap();
/// let r = vec![1.0, 2.0];
/// let z = precond.apply(&r).unwrap();
/// // z[i] ≈ r[i] / a[i,i]
/// assert!((z[0] - 0.25_f64).abs() < 1e-10);
/// assert!((z[1] - 0.5_f64).abs() < 1e-10);
/// ```
pub struct JacobiPreconditioner<T: Float> {
    diag_inv: Vec<T>,
}

impl<T: Float> JacobiPreconditioner<T> {
    /// Create Jacobi preconditioner from diagonal of A
    ///
    /// # Complexity
    ///
    /// O(nnz) time to extract diagonal
    pub fn from_matrix(a: &CsrMatrix<T>) -> SparseResult<Self> {
        let n = a.nrows();
        let mut diag_inv = vec![T::zero(); n];

        // Extract diagonal elements
        for (i, diag_val) in diag_inv.iter_mut().enumerate().take(n) {
            let row_start = a.row_ptr()[i];
            let row_end = a.row_ptr()[i + 1];

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                if i == j {
                    let val = a.values()[idx];
                    if val.abs() < T::epsilon() {
                        return Err(SparseError::operation(&format!(
                            "Zero diagonal element at index {}",
                            i
                        )));
                    }
                    *diag_val = T::one() / val;
                    break;
                }
            }

            // Check if diagonal was found
            if *diag_val == T::zero() {
                return Err(SparseError::operation(&format!(
                    "Missing diagonal element at index {}",
                    i
                )));
            }
        }

        Ok(Self { diag_inv })
    }
}

impl<T: Float> Preconditioner<T> for JacobiPreconditioner<T> {
    fn apply(&self, r: &[T]) -> SparseResult<Vec<T>> {
        // M^{-1} * r = diag(A)^{-1} * r
        Ok(r.iter()
            .zip(self.diag_inv.iter())
            .map(|(ri, di)| *ri * *di)
            .collect())
    }
}

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner
///
/// Uses M = (D + ωL) D^{-1} (D + ωU) as the preconditioner, where:
/// - D is the diagonal of A
/// - L is the strictly lower triangular part of A
/// - U is the strictly upper triangular part of A
/// - ω is the relaxation parameter (typically 1.0)
///
/// SSOR is effective for symmetric matrices and provides better convergence
/// than Jacobi for many problems.
///
/// # Examples
///
/// ```rust
/// use tenrso_sparse::{CsrMatrix, solvers::{SsorPreconditioner, Preconditioner}};
///
/// let row_ptr = vec![0, 2, 4];
/// let col_indices = vec![0, 1, 0, 1];
/// let values = vec![4.0, -1.0, -1.0, 4.0];
/// let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
///
/// let precond = SsorPreconditioner::from_matrix(&a, 1.0).unwrap();
/// let r = vec![1.0, 2.0];
/// let z = precond.apply(&r).unwrap();
/// assert!(z.len() == 2);
/// ```
pub struct SsorPreconditioner<T: Float> {
    a: CsrMatrix<T>,
    omega: T,
    diag_inv: Vec<T>,
}

impl<T: Float> SsorPreconditioner<T> {
    /// Create SSOR preconditioner with relaxation parameter omega
    ///
    /// # Arguments
    ///
    /// - `a`: Sparse matrix (should be symmetric for best results)
    /// - `omega`: Relaxation parameter (typically 1.0, must be in (0, 2))
    ///
    /// # Complexity
    ///
    /// O(nnz) time to extract diagonal
    pub fn from_matrix(a: &CsrMatrix<T>, omega: T) -> SparseResult<Self> {
        let two = T::from(2.0).ok_or_else(|| {
            SparseError::operation("SSOR: failed to convert 2.0 to target float type")
        })?;
        if omega <= T::zero() || omega >= two {
            return Err(SparseError::validation("SSOR omega must be in (0, 2)"));
        }

        let n = a.nrows();
        let mut diag_inv = vec![T::zero(); n];

        // Extract diagonal elements
        for (i, diag_val) in diag_inv.iter_mut().enumerate().take(n) {
            let row_start = a.row_ptr()[i];
            let row_end = a.row_ptr()[i + 1];

            for idx in row_start..row_end {
                let j = a.col_indices()[idx];
                if i == j {
                    let val = a.values()[idx];
                    if val.abs() < T::epsilon() {
                        return Err(SparseError::operation(&format!(
                            "Zero diagonal element at index {}",
                            i
                        )));
                    }
                    *diag_val = T::one() / val;
                    break;
                }
            }

            if *diag_val == T::zero() {
                return Err(SparseError::operation(&format!(
                    "Missing diagonal element at index {}",
                    i
                )));
            }
        }

        Ok(Self {
            a: a.clone(),
            omega,
            diag_inv,
        })
    }
}

impl<T: Float> Preconditioner<T> for SsorPreconditioner<T> {
    fn apply(&self, r: &[T]) -> SparseResult<Vec<T>> {
        let n = self.a.nrows();
        let mut y = vec![T::zero(); n];
        let mut z = vec![T::zero(); n];

        // Forward sweep: solve (D + ωL)y = ωr
        for i in 0..n {
            let mut sum = T::zero();
            let row_start = self.a.row_ptr()[i];
            let row_end = self.a.row_ptr()[i + 1];

            // Sum over strictly lower triangular part
            for idx in row_start..row_end {
                let j = self.a.col_indices()[idx];
                if j < i {
                    sum = sum + self.a.values()[idx] * y[j];
                }
            }

            y[i] = (self.omega * r[i] - self.omega * sum) * self.diag_inv[i];
        }

        // Backward sweep: solve (D + ωU)z = Dy
        for i in (0..n).rev() {
            let mut sum = T::zero();
            let row_start = self.a.row_ptr()[i];
            let row_end = self.a.row_ptr()[i + 1];

            // Sum over strictly upper triangular part
            for idx in row_start..row_end {
                let j = self.a.col_indices()[idx];
                if j > i {
                    sum = sum + self.a.values()[idx] * z[j];
                }
            }

            let diag = T::one() / self.diag_inv[i];
            z[i] = (diag * y[i] - self.omega * sum) * self.diag_inv[i];
        }

        Ok(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_preconditioner() {
        let precond = IdentityPreconditioner;
        let r = vec![1.0, 2.0, 3.0];
        let z = precond.apply(&r).unwrap();
        assert_eq!(z, r);
    }

    #[test]
    fn test_jacobi_preconditioner() {
        // A = [[4, -1], [-1, 4]] (diagonally dominant)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let precond = JacobiPreconditioner::from_matrix(&a).unwrap();

        // Test application
        let r = vec![1.0, 2.0];
        let z = precond.apply(&r).unwrap();

        // z[i] = r[i] / a[i,i]
        assert!((z[0] - 0.25).abs() < 1e-10);
        assert!((z[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_preconditioner_zero_diagonal() {
        // Matrix with zero diagonal element
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![1, 0];
        let values = vec![1.0, 1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let result = JacobiPreconditioner::<f64>::from_matrix(&a);
        assert!(result.is_err());
    }

    #[test]
    fn test_ssor_preconditioner() {
        // A = [[4, -1], [-1, 4]] (symmetric)
        let row_ptr = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![4.0, -1.0, -1.0, 4.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        let precond = SsorPreconditioner::from_matrix(&a, 1.0).unwrap();

        // Test application
        let r = vec![1.0, 2.0];
        let z = precond.apply(&r).unwrap();
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn test_ssor_invalid_omega() {
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0, 1.0];
        let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();

        // omega = 0 (invalid)
        let result = SsorPreconditioner::from_matrix(&a, 0.0);
        assert!(result.is_err());

        // omega = 2.0 (invalid)
        let result = SsorPreconditioner::from_matrix(&a, 2.0);
        assert!(result.is_err());

        // omega = 1.0 (valid)
        let result = SsorPreconditioner::from_matrix(&a, 1.0);
        assert!(result.is_ok());
    }
}
