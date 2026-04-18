//! Iterative linear solvers for sparse systems
//!
//! This module provides iterative methods for solving sparse linear systems Ax = b:
//! - Conjugate Gradient (CG) for symmetric positive definite matrices
//! - BiCGSTAB for general nonsymmetric matrices
//! - GMRES for general nonsymmetric matrices (restarted version)
//! - MINRES for symmetric (possibly indefinite) matrices
//! - CGNE (CG on Normal Equations) for least squares problems (overdetermined systems)
//! - CGNR (CG on Normal Residual) for least squares problems (underdetermined systems)
//!
//! All solvers support preconditioning using ILU/IC factorizations.
//!
//! # Examples
//!
//! ```rust
//! use tenrso_sparse::{CsrMatrix, solvers, solvers::IdentityPreconditioner};
//!
//! // Create a simple SPD system: A = [[2, -1], [-1, 2]]
//! let row_ptr = vec![0, 2, 4];
//! let col_indices = vec![0, 1, 0, 1];
//! let values = vec![2.0, -1.0, -1.0, 2.0];
//! let a = CsrMatrix::new(row_ptr, col_indices, values, (2, 2)).unwrap();
//!
//! // Right-hand side
//! let b = vec![1.0, 1.0];
//!
//! // Solve using CG
//! let result = solvers::cg::<f64, IdentityPreconditioner>(&a, &b, 100, 1e-6, None);
//! assert!(result.is_ok());
//! ```

use std::fmt;

pub mod bicgstab;
pub mod cg;
pub mod cgne;
pub mod cgnr;
pub mod gmres;
mod helpers;
pub mod minres;
pub mod preconditioner;

pub use bicgstab::bicgstab;
pub use cg::cg;
pub use cgne::cgne;
pub use cgnr::cgnr;
pub use gmres::gmres;
pub use minres::minres;
pub use preconditioner::{
    IdentityPreconditioner, IluPreconditioner, JacobiPreconditioner, Preconditioner,
    SsorPreconditioner,
};

/// Solver convergence information
#[derive(Debug, Clone)]
pub struct SolverInfo {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
    /// Whether the solver converged
    pub converged: bool,
}

impl fmt::Display for SolverInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Solver: {} in {} iterations, residual = {:.2e}",
            if self.converged {
                "converged"
            } else {
                "did not converge"
            },
            self.iterations,
            self.residual
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_info_display() {
        let info = SolverInfo {
            iterations: 10,
            residual: 1.5e-7,
            converged: true,
        };

        let s = format!("{}", info);
        assert!(s.contains("converged"));
        assert!(s.contains("10"));
    }
}
