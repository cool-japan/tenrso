//! Type definitions for CP decomposition
//!
//! Contains all public types: CpError, InitStrategy, CpConstraints,
//! ConvergenceReason, ConvergenceInfo, CpDecomp, IncrementalMode,
//! and RegularizationType.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::numeric::{Float, FloatConst, NumCast};
use tenrso_core::DenseND;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CpError {
    #[error("Invalid rank: {0}")]
    InvalidRank(usize),

    #[error("Invalid tolerance: {0}")]
    InvalidTolerance(f64),

    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] scirs2_linalg::LinalgError),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),

    #[error("Non-negative constraint violated: input tensor contains negative values")]
    NonnegativeViolation,

    #[error("Invalid regularization parameter: {0}")]
    InvalidRegularization(String),
}

/// Initialization strategy for CP-ALS
#[derive(Debug, Clone, Copy)]
pub enum InitStrategy {
    /// Random initialization from uniform distribution [0, 1]
    Random,
    /// Random initialization from normal distribution N(0, 1)
    RandomNormal,
    /// SVD-based initialization (HOSVD)
    Svd,
    /// Non-negative SVD initialization (NNSVD)
    ///
    /// Based on Boutsidis & Gallopoulos (2008).
    /// Uses SVD with non-negativity constraints, suitable for
    /// non-negative decompositions (e.g., topic modeling, NMF-style).
    Nnsvd,
    /// Leverage score sampling initialization
    ///
    /// Based on statistical leverage scores from SVD.
    /// Samples important rows/columns based on their contribution
    /// to the low-rank approximation. More principled than random
    /// initialization for large-scale tensors.
    LeverageScore,
}

/// Regularization type for CP-ALS decomposition
///
/// Controls the type and strength of regularization applied to factor matrices.
#[derive(Debug, Clone, Copy)]
pub enum RegularizationType {
    /// No regularization
    None,

    /// L2 (Ridge / Tikhonov) regularization: adds lambda * ||F||_F^2 penalty
    ///
    /// Modifies the normal equations by adding lambda * I to the Gram matrix.
    /// This shrinks factor values toward zero, preventing overfitting
    /// and improving numerical stability.
    L2 {
        /// Regularization strength (lambda >= 0)
        lambda: f64,
    },

    /// L1 (Lasso) regularization: adds lambda * ||F||_1 penalty
    ///
    /// Promotes sparsity in factor matrices via soft-thresholding.
    /// After each ALS update, elements are shrunk toward zero:
    /// f_ij = sign(f_ij) * max(|f_ij| - lambda, 0)
    L1 {
        /// Regularization strength (lambda >= 0)
        lambda: f64,
    },

    /// Elastic net: combines L1 and L2 regularization
    ///
    /// Penalty = alpha * lambda * ||F||_1 + (1 - alpha) * lambda * ||F||_F^2
    /// where alpha in [0, 1] controls the L1/L2 mix.
    ElasticNet {
        /// Overall regularization strength (lambda >= 0)
        lambda: f64,
        /// L1 ratio: alpha in [0, 1], where 1 = pure L1, 0 = pure L2
        alpha: f64,
    },

    /// Tikhonov regularization with a custom regularization matrix
    ///
    /// Adds lambda * ||Gamma * F||_F^2 penalty where Gamma is a
    /// regularization matrix (e.g., finite difference operator for smoothness).
    /// When Gamma = I, this reduces to standard L2 regularization.
    Tikhonov {
        /// Regularization strength (lambda >= 0)
        lambda: f64,
        /// Order of the finite difference operator (0 = identity/ridge, 1 = first-order, 2 = second-order)
        order: usize,
    },
}

impl RegularizationType {
    /// Validate the regularization parameters
    pub fn validate(&self) -> Result<(), CpError> {
        match self {
            RegularizationType::None => Ok(()),
            RegularizationType::L2 { lambda } => {
                if *lambda < 0.0 {
                    Err(CpError::InvalidRegularization(format!(
                        "L2 lambda must be >= 0, got {}",
                        lambda
                    )))
                } else {
                    Ok(())
                }
            }
            RegularizationType::L1 { lambda } => {
                if *lambda < 0.0 {
                    Err(CpError::InvalidRegularization(format!(
                        "L1 lambda must be >= 0, got {}",
                        lambda
                    )))
                } else {
                    Ok(())
                }
            }
            RegularizationType::ElasticNet { lambda, alpha } => {
                if *lambda < 0.0 {
                    Err(CpError::InvalidRegularization(format!(
                        "ElasticNet lambda must be >= 0, got {}",
                        lambda
                    )))
                } else if !(0.0..=1.0).contains(alpha) {
                    Err(CpError::InvalidRegularization(format!(
                        "ElasticNet alpha must be in [0, 1], got {}",
                        alpha
                    )))
                } else {
                    Ok(())
                }
            }
            RegularizationType::Tikhonov { lambda, .. } => {
                if *lambda < 0.0 {
                    Err(CpError::InvalidRegularization(format!(
                        "Tikhonov lambda must be >= 0, got {}",
                        lambda
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Get the L2 component of this regularization for Gram matrix modification
    pub fn l2_component(&self) -> f64 {
        match self {
            RegularizationType::None => 0.0,
            RegularizationType::L2 { lambda } => *lambda,
            RegularizationType::L1 { .. } => 0.0,
            RegularizationType::ElasticNet { lambda, alpha } => (1.0 - alpha) * lambda,
            RegularizationType::Tikhonov { lambda, order } => {
                if *order == 0 {
                    *lambda
                } else {
                    0.0
                }
            }
        }
    }

    /// Get the L1 component of this regularization for soft-thresholding
    pub fn l1_component(&self) -> f64 {
        match self {
            RegularizationType::None => 0.0,
            RegularizationType::L2 { .. } => 0.0,
            RegularizationType::L1 { lambda } => *lambda,
            RegularizationType::ElasticNet { lambda, alpha } => alpha * lambda,
            RegularizationType::Tikhonov { .. } => 0.0,
        }
    }
}

/// Constraints for CP-ALS decomposition
///
/// Allows control over factor matrix properties during optimization
#[derive(Debug, Clone, Copy)]
pub struct CpConstraints {
    /// Enforce non-negativity on all factor matrices
    /// When true, negative values are projected to zero after each update
    pub nonnegative: bool,

    /// L2 regularization parameter (lambda >= 0)
    /// Adds lambda * ||F||^2 penalty to prevent overfitting
    /// Set to 0.0 to disable regularization
    ///
    /// Note: For advanced regularization (L1, elastic net, Tikhonov),
    /// use the `regularization` field instead. If both `l2_reg` and
    /// `regularization` are set, `regularization` takes precedence.
    pub l2_reg: f64,

    /// Enforce orthogonality constraints on factor matrices
    /// When true, factors are orthonormalized after each update
    /// Note: This may conflict with non-negativity constraints
    pub orthogonal: bool,

    /// Advanced regularization type
    /// When set to anything other than None, overrides `l2_reg`.
    pub regularization: RegularizationType,

    /// Whether to validate that the input tensor is non-negative
    /// when the nonnegative constraint is active. Default: true.
    pub validate_nonneg_input: bool,
}

impl Default for CpConstraints {
    fn default() -> Self {
        Self {
            nonnegative: false,
            l2_reg: 0.0,
            orthogonal: false,
            regularization: RegularizationType::None,
            validate_nonneg_input: true,
        }
    }
}

impl CpConstraints {
    /// Create constraints with non-negativity enforcement
    pub fn nonnegative() -> Self {
        Self {
            nonnegative: true,
            ..Default::default()
        }
    }

    /// Create constraints with L2 regularization
    pub fn l2_regularized(lambda: f64) -> Self {
        Self {
            l2_reg: lambda,
            regularization: RegularizationType::L2 { lambda },
            ..Default::default()
        }
    }

    /// Create constraints with L1 (sparsity-promoting) regularization
    pub fn l1_regularized(lambda: f64) -> Self {
        Self {
            regularization: RegularizationType::L1 { lambda },
            ..Default::default()
        }
    }

    /// Create constraints with elastic net regularization
    pub fn elastic_net(lambda: f64, alpha: f64) -> Self {
        Self {
            regularization: RegularizationType::ElasticNet { lambda, alpha },
            ..Default::default()
        }
    }

    /// Create constraints with Tikhonov regularization
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength
    /// * `order` - Order of finite difference operator (0 = ridge, 1 = first-order smoothness, 2 = second-order)
    pub fn tikhonov(lambda: f64, order: usize) -> Self {
        Self {
            regularization: RegularizationType::Tikhonov { lambda, order },
            ..Default::default()
        }
    }

    /// Create constraints with orthogonality enforcement
    pub fn orthogonal() -> Self {
        Self {
            orthogonal: true,
            ..Default::default()
        }
    }

    /// Get the effective L2 regularization parameter
    ///
    /// Returns the L2 component from `regularization` if set,
    /// otherwise falls back to `l2_reg`.
    pub fn effective_l2(&self) -> f64 {
        match self.regularization {
            RegularizationType::None => self.l2_reg,
            _ => self.regularization.l2_component(),
        }
    }

    /// Get the effective L1 regularization parameter
    pub fn effective_l1(&self) -> f64 {
        self.regularization.l1_component()
    }
}

/// Convergence reason for decomposition algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceReason {
    /// Converged: fit change below tolerance
    FitTolerance,
    /// Reached maximum iterations
    MaxIterations,
    /// Detected oscillation in fit values
    Oscillation,
    /// Time limit exceeded (if applicable)
    TimeLimit,
}

/// Convergence diagnostics for decomposition algorithms
///
/// Tracks detailed convergence information including fit history,
/// oscillation detection, and convergence reason.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo<T> {
    /// History of fit values at each iteration
    pub fit_history: Vec<T>,

    /// Final convergence reason
    pub reason: ConvergenceReason,

    /// Whether oscillation was detected
    pub oscillated: bool,

    /// Number of oscillations detected (fit increased instead of decreased)
    pub oscillation_count: usize,

    /// Final relative fit change
    pub final_fit_change: T,
}

/// CP decomposition result
///
/// Represents a tensor as a sum of R rank-1 tensors.
#[derive(Debug, Clone)]
pub struct CpDecomp<T> {
    /// Factor matrices, one for each mode
    /// Each matrix has shape (In, R) where In is the mode size and R is the rank
    pub factors: Vec<Array2<T>>,

    /// Weights for each rank-1 component (optional)
    /// If None, weights are absorbed into the factor matrices
    pub weights: Option<Array1<T>>,

    /// Final fit value (normalized reconstruction error)
    /// fit = 1 - ||X - X_reconstructed|| / ||X||
    pub fit: T,

    /// Number of iterations performed
    pub iters: usize,

    /// Convergence diagnostics (if enabled)
    pub convergence: Option<ConvergenceInfo<T>>,
}

impl<T> CpDecomp<T>
where
    T: Float + FloatConst + NumCast,
{
    /// Reconstruct the original tensor from the CP decomposition
    ///
    /// Computes X ~ Sum_r lambda_r (u1r x u2r x ... x unr)
    ///
    /// Uses optimized CP reconstruction from tenrso-kernels.
    ///
    /// # Complexity
    ///
    /// Time: O(R * prod_i I_i)
    /// Space: O(prod_i I_i)
    pub fn reconstruct(&self, shape: &[usize]) -> Result<DenseND<T>> {
        let n_modes = self.factors.len();

        // Verify shape compatibility
        if n_modes != shape.len() {
            anyhow::bail!(
                "Shape rank mismatch: expected {} modes, got {}",
                n_modes,
                shape.len()
            );
        }

        for (i, factor) in self.factors.iter().enumerate() {
            if factor.shape()[0] != shape[i] {
                anyhow::bail!(
                    "Mode-{} size mismatch: expected {}, got {}",
                    i,
                    shape[i],
                    factor.shape()[0]
                );
            }
        }

        // Use optimized kernel reconstruction
        let factor_views: Vec<_> = self.factors.iter().map(|f| f.view()).collect();
        let weights_view = self.weights.as_ref().map(|w| w.view());

        let reconstructed = tenrso_kernels::cp_reconstruct(&factor_views, weights_view.as_ref())?;

        // Wrap in DenseND
        Ok(DenseND::from_array(reconstructed))
    }

    /// Extract weights from factor matrices by normalizing columns
    ///
    /// Each factor matrix column is normalized to unit length,
    /// and the norms are accumulated as weights.
    pub fn extract_weights(&mut self) {
        let rank = self.factors[0].shape()[1];
        let mut weights = Array1::<T>::ones(rank);

        for factor in &mut self.factors {
            for r in 0..rank {
                let mut norm_sq = T::zero();
                for i in 0..factor.shape()[0] {
                    let val = factor[[i, r]];
                    norm_sq = norm_sq + val * val;
                }

                let norm = norm_sq.sqrt();
                if norm > T::epsilon() {
                    weights[r] = weights[r] * norm;

                    // Normalize column
                    for i in 0..factor.shape()[0] {
                        factor[[i, r]] = factor[[i, r]] / norm;
                    }
                }
            }
        }

        self.weights = Some(weights);
    }
}

/// Update mode for incremental CP-ALS
#[derive(Debug, Clone, Copy)]
pub enum IncrementalMode {
    /// Append new data (grow the tensor in one mode)
    /// New data is concatenated along the specified mode
    Append,

    /// Sliding window (maintain tensor size)
    /// Old data is discarded, new data replaces it
    SlidingWindow {
        /// Forgetting factor lambda in (0, 1]
        /// lambda=1: equal weight to all data
        /// lambda<1: exponentially forget old data
        lambda: f64,
    },
}
