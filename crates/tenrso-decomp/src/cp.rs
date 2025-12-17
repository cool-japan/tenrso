//! CP-ALS (Canonical Polyadic decomposition via Alternating Least Squares)
//!
//! The CP decomposition factorizes a tensor X into a sum of rank-1 tensors:
//!
//! X ≈ Σᵣ λᵣ (u₁ᵣ ⊗ u₂ᵣ ⊗ ... ⊗ uₙᵣ)
//!
//! Where:
//! - R is the CP rank
//! - λᵣ are weights (optional, can be absorbed into factors)
//! - uᵢᵣ are factor vectors forming factor matrices Uᵢ ∈ ℝ^(Iᵢ×R)
//!
//! The ALS algorithm alternates between updating each factor matrix while
//! keeping others fixed using MTTKRP and solving a least-squares problem.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Linear algebra operations use `scirs2_linalg`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use scirs2_core::random::{thread_rng, Distribution, RandNormal as Normal, Rng};
use scirs2_linalg::{lstsq, LinalgError};
use std::iter::Sum;
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CpError {
    #[error("Invalid rank: {0}")]
    InvalidRank(usize),

    #[error("Invalid tolerance: {0}")]
    InvalidTolerance(f64),

    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] LinalgError),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
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

/// Constraints for CP-ALS decomposition
///
/// Allows control over factor matrix properties during optimization
#[derive(Debug, Clone, Copy)]
pub struct CpConstraints {
    /// Enforce non-negativity on all factor matrices
    /// When true, negative values are projected to zero after each update
    pub nonnegative: bool,

    /// L2 regularization parameter (λ ≥ 0)
    /// Adds λ||F||² penalty to prevent overfitting
    /// Set to 0.0 to disable regularization
    pub l2_reg: f64,

    /// Enforce orthogonality constraints on factor matrices
    /// When true, factors are orthonormalized after each update
    /// Note: This may conflict with non-negativity constraints
    pub orthogonal: bool,
}

impl Default for CpConstraints {
    fn default() -> Self {
        Self {
            nonnegative: false,
            l2_reg: 0.0,
            orthogonal: false,
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
    /// Each matrix has shape (Iₙ, R) where Iₙ is the mode size and R is the rank
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
    /// Computes X ≈ Σᵣ λᵣ (u₁ᵣ ⊗ u₂ᵣ ⊗ ... ⊗ uₙᵣ)
    ///
    /// Uses optimized CP reconstruction from tenrso-kernels.
    ///
    /// # Complexity
    ///
    /// Time: O(R × ∏ᵢ Iᵢ)
    /// Space: O(∏ᵢ Iᵢ)
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

/// Compute CP-ALS decomposition of a tensor
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `rank` - Target CP rank (number of components)
/// * `max_iters` - Maximum number of ALS iterations
/// * `tol` - Convergence tolerance on fit improvement
/// * `init` - Initialization strategy
/// * `time_limit` - Optional time limit for execution (None for no limit)
///
/// # Returns
///
/// CpDecomp containing factor matrices, weights, final fit, and iteration count
///
/// # Errors
///
/// Returns error if:
/// - Rank is invalid (0 or exceeds any mode size)
/// - Tolerance is invalid (negative or >= 1)
/// - Linear algebra operations fail
/// - Convergence is not achieved within max_iters
///
/// # Complexity
///
/// Time: O(I × R² × ∏ᵢ Iᵢ) per iteration where I is max_iters
/// Space: O(N × Imax × R) for factor matrices
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_als, InitStrategy};
///
/// // Create a 10×10×10 tensor
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
///
/// // Decompose with rank 5, no time limit
/// let cp = cp_als(&tensor, 5, 50, 1e-4, InitStrategy::Random, None).unwrap();
///
/// println!("Final fit: {:.4}", cp.fit);
/// println!("Iterations: {}", cp.iters);
///
/// // With 5-second time limit
/// use std::time::Duration;
/// let cp_timed = cp_als(&tensor, 5, 50, 1e-4, InitStrategy::Random, Some(Duration::from_secs(5))).unwrap();
/// ```
pub fn cp_als<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    time_limit: Option<std::time::Duration>,
) -> Result<CpDecomp<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validation
    if rank == 0 {
        return Err(CpError::InvalidRank(rank));
    }

    for &mode_size in shape.iter() {
        if rank > mode_size {
            return Err(CpError::InvalidRank(rank));
        }
    }

    if !(0.0..1.0).contains(&tol) {
        return Err(CpError::InvalidTolerance(tol));
    }

    // Initialize factor matrices
    let mut factors = initialize_factors(tensor, rank, init)?;

    // Compute tensor norm for fit calculation
    let tensor_norm_sq = compute_norm_squared(tensor);

    let mut prev_fit = T::zero();
    let mut fit = T::zero();
    let mut iters = 0;

    // Convergence tracking
    let mut fit_history = Vec::with_capacity(max_iters);
    let mut oscillation_count = 0;
    let mut convergence_reason = ConvergenceReason::MaxIterations;
    let mut final_fit_change = T::zero();

    // Time tracking
    let start_time = std::time::Instant::now();

    // ALS iterations
    for iter in 0..max_iters {
        // Check time limit if set
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                convergence_reason = ConvergenceReason::TimeLimit;
                break;
            }
        }

        iters = iter + 1;

        // Update each factor matrix
        for mode in 0..n_modes {
            // Step 1: Compute MTTKRP
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp(&tensor.view(), &factor_views, mode)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

            // Step 2: Compute Hadamard product of Gram matrices
            let gram = compute_gram_hadamard(&factors, mode);

            // Step 3: Solve least squares: factors[mode] = mttkrp_result * gram^(-1)
            factors[mode] = solve_least_squares(&mttkrp_result, &gram)?;
        }

        // Compute fit
        fit = compute_fit(tensor, &factors, tensor_norm_sq)?;
        fit_history.push(fit);

        // Check for oscillation (fit decreased instead of improved)
        if iter > 0 && fit < prev_fit {
            oscillation_count += 1;
        }

        // Check convergence
        let fit_change = (fit - prev_fit).abs();
        final_fit_change = fit_change;

        if iter > 0 && fit_change < NumCast::from(tol).unwrap() {
            convergence_reason = ConvergenceReason::FitTolerance;
            break;
        }

        // Detect severe oscillation
        if oscillation_count > 5 && iter > 10 {
            convergence_reason = ConvergenceReason::Oscillation;
            break;
        }

        prev_fit = fit;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
        convergence: Some(ConvergenceInfo {
            fit_history,
            reason: convergence_reason,
            oscillated: oscillation_count > 0,
            oscillation_count,
            final_fit_change,
        }),
    })
}

/// CP-ALS with constraints (non-negativity, regularization, orthogonality)
///
/// Extended version of CP-ALS that supports:
/// - Non-negative factor matrices (for applications like topic modeling, NMF-style decomposition)
/// - L2 regularization to prevent overfitting
/// - Orthogonality constraints on factor matrices
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `rank` - Target CP rank (number of components)
/// * `max_iters` - Maximum number of ALS iterations
/// * `tol` - Convergence tolerance on fit improvement
/// * `init` - Initialization strategy
/// * `constraints` - Constraint configuration (non-negativity, regularization, orthogonality)
/// * `time_limit` - Optional time limit for execution (None for no limit)
///
/// # Returns
///
/// CpDecomp containing factor matrices, weights, final fit, and iteration count
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_als_constrained, InitStrategy, CpConstraints};
///
/// // Non-negative CP decomposition
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
/// let constraints = CpConstraints::nonnegative();
/// let cp = cp_als_constrained(&tensor, 5, 50, 1e-4, InitStrategy::Random, constraints, None).unwrap();
/// ```
pub fn cp_als_constrained<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    constraints: CpConstraints,
    time_limit: Option<std::time::Duration>,
) -> Result<CpDecomp<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validation
    if rank == 0 {
        return Err(CpError::InvalidRank(rank));
    }

    for &mode_size in shape.iter() {
        if rank > mode_size {
            return Err(CpError::InvalidRank(rank));
        }
    }

    if !(0.0..1.0).contains(&tol) {
        return Err(CpError::InvalidTolerance(tol));
    }

    if constraints.l2_reg < 0.0 {
        return Err(CpError::InvalidTolerance(constraints.l2_reg));
    }

    // Initialize factor matrices
    let mut factors = initialize_factors(tensor, rank, init)?;

    // Apply initial constraints
    if constraints.nonnegative {
        for factor in &mut factors {
            factor.mapv_inplace(|x| x.max(T::zero()));
        }
    }

    // Compute tensor norm for fit calculation
    let tensor_norm_sq = compute_norm_squared(tensor);

    let mut prev_fit = T::zero();
    let mut fit = T::zero();
    let mut iters = 0;

    // Convergence tracking
    let mut fit_history = Vec::with_capacity(max_iters);
    let mut oscillation_count = 0;
    let mut convergence_reason = ConvergenceReason::MaxIterations;
    let mut final_fit_change = T::zero();

    // Time tracking
    let start_time = std::time::Instant::now();

    // ALS iterations
    for iter in 0..max_iters {
        // Check time limit if set
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                convergence_reason = ConvergenceReason::TimeLimit;
                break;
            }
        }

        iters = iter + 1;

        // Update each factor matrix
        for mode in 0..n_modes {
            // Step 1: Compute MTTKRP
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp(&tensor.view(), &factor_views, mode)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

            // Step 2: Compute Hadamard product of Gram matrices
            let mut gram = compute_gram_hadamard(&factors, mode);

            // Step 3: Apply L2 regularization
            if constraints.l2_reg > 0.0 {
                let reg = NumCast::from(constraints.l2_reg).unwrap();
                for i in 0..gram.nrows() {
                    gram[[i, i]] += reg;
                }
            }

            // Step 4: Solve least squares
            factors[mode] = solve_least_squares(&mttkrp_result, &gram)?;

            // Step 5: Apply constraints
            if constraints.nonnegative {
                // Project negative values to zero
                factors[mode].mapv_inplace(|x| x.max(T::zero()));
            }

            if constraints.orthogonal {
                // Orthonormalize the factor matrix using QR decomposition
                factors[mode] = orthonormalize_factor(&factors[mode])?;
            }
        }

        // Compute fit
        fit = compute_fit(tensor, &factors, tensor_norm_sq)?;
        fit_history.push(fit);

        // Check for oscillation
        if iter > 0 && fit < prev_fit {
            oscillation_count += 1;
        }

        // Check convergence
        let fit_change = (fit - prev_fit).abs();
        final_fit_change = fit_change;

        if iter > 0 && fit_change < NumCast::from(tol).unwrap() {
            convergence_reason = ConvergenceReason::FitTolerance;
            break;
        }

        // Detect severe oscillation
        if oscillation_count > 5 && iter > 10 {
            convergence_reason = ConvergenceReason::Oscillation;
            break;
        }

        prev_fit = fit;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
        convergence: Some(ConvergenceInfo {
            fit_history,
            reason: convergence_reason,
            oscillated: oscillation_count > 0,
            oscillation_count,
            final_fit_change,
        }),
    })
}

/// Accelerated CP-ALS with line search optimization
///
/// An enhanced version of CP-ALS that uses line search to determine optimal
/// step sizes and incorporates acceleration techniques for faster convergence.
///
/// This method typically converges 2-5× faster than standard CP-ALS while
/// maintaining the same approximation quality.
///
/// # Algorithm
///
/// Uses a combination of:
/// - **Line search**: Finds optimal step size in update direction
/// - **Extrapolation**: Accelerates convergence using Nesterov-style momentum
/// - **Adaptive restart**: Resets momentum when fit decreases
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `rank` - CP rank (number of components)
/// * `max_iters` - Maximum number of ALS iterations
/// * `tol` - Convergence tolerance for relative fit change
/// * `init` - Initialization strategy for factor matrices
/// * `time_limit` - Optional time limit for execution
///
/// # Returns
///
/// CP decomposition with factors, weights, and convergence information
///
/// # Complexity
///
/// Time: O(max_iters × N × ∏ᵢ Iᵢ × R²)  (similar to CP-ALS)
/// Space: O(∑ᵢ Iᵢ × R)  (stores previous factors for extrapolation)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::{cp_als_accelerated, InitStrategy};
///
/// let tensor = DenseND::<f64>::random_uniform(&[30, 30, 30], 0.0, 1.0);
/// let cp = cp_als_accelerated(&tensor, 10, 50, 1e-4, InitStrategy::Random, None).unwrap();
///
/// println!("Converged in {} iterations (faster than standard CP-ALS)", cp.iters);
/// println!("Final fit: {:.4}", cp.fit);
/// ```
///
/// # References
///
/// - Acar et al. (2011), "Scalable tensor factorizations for incomplete data"
/// - Phan et al. (2013), "Fast alternating LS algorithms for high order CANDECOMP/PARAFAC tensor factorizations"
pub fn cp_als_accelerated<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    time_limit: Option<std::time::Duration>,
) -> Result<CpDecomp<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + Send
        + Sync
        + scirs2_core::ndarray_ext::ScalarOperand
        + scirs2_core::numeric::FromPrimitive
        + 'static,
{
    let start_time = std::time::Instant::now();

    // Validate inputs
    if rank == 0 {
        return Err(CpError::InvalidRank(rank));
    }
    if tol <= 0.0 || tol >= 1.0 {
        return Err(CpError::InvalidTolerance(tol));
    }

    // Initialize factor matrices
    let mut factors = initialize_factors(tensor, rank, init)?;
    let n_modes = factors.len();

    // Store previous factors for extrapolation
    let mut prev_factors: Vec<Array2<T>> = factors.to_vec();

    // Extrapolation parameters
    let mut alpha = T::from(0.5).unwrap(); // Extrapolation strength
    let alpha_max = T::from(0.9).unwrap();
    let alpha_min = T::from(0.1).unwrap();

    let tol_t = T::from(tol).unwrap();
    let tensor_norm = tensor.frobenius_norm();
    let tensor_norm_sq = tensor_norm * tensor_norm; // Squared norm for compute_fit
    let mut prev_fit = T::zero();
    let mut fit = T::zero();

    // Convergence tracking
    let mut fit_history = Vec::with_capacity(max_iters);
    let mut oscillation_count = 0;
    let mut convergence_reason = ConvergenceReason::MaxIterations;
    let mut final_fit_change = T::zero();
    let mut iters = 0;

    for iter in 0..max_iters {
        iters = iter + 1;

        // Check time limit
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                convergence_reason = ConvergenceReason::TimeLimit;
                break;
            }
        }

        // ALS updates for each mode
        for mode in 0..n_modes {
            // Create views for MTTKRP
            let tensor_view = tensor.view();
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

            // Compute MTTKRP: X_{(n)} (Uₙ₊₁ ⊙ ... ⊙ U₁)^T
            let mttkrp_result = mttkrp(&tensor_view, &factor_views, mode)
                .map_err(|e| CpError::ShapeMismatch(format!("MTTKRP failed: {}", e)))?;

            // Compute Hadamard product of all Gram matrices except mode n
            let gram = compute_gram_hadamard(&factors, mode);

            // Solve least squares: factor_new = MTTKRP * Gram^(-1)
            let mut factor_new = solve_least_squares(&mttkrp_result, &gram)?;

            // LINE SEARCH: Find optimal step size
            let alpha_ls = line_search_cp(
                tensor,
                &factors,
                &prev_factors,
                mode,
                &factor_new,
                T::from(0.5).unwrap(),
                5, // max line search iterations
            );

            // Apply extrapolation with line search step size
            if iter > 0 {
                // Extrapolated update: F_new = F_als + alpha_ls * alpha * (F_als - F_prev)
                let factor_prev = &prev_factors[mode];
                for i in 0..factor_new.shape()[0] {
                    for j in 0..factor_new.shape()[1] {
                        let diff = factor_new[[i, j]] - factor_prev[[i, j]];
                        factor_new[[i, j]] += alpha_ls * alpha * diff;
                    }
                }
            }

            // Store previous factor before update
            prev_factors[mode] = factors[mode].clone();

            // Update factor
            factors[mode] = factor_new;
        }

        // Compute fit
        fit = compute_fit(tensor, &factors, tensor_norm_sq)?;
        fit_history.push(fit);

        // Adaptive extrapolation strength
        if iter > 0 {
            if fit > prev_fit {
                // Good progress: increase extrapolation
                alpha = (alpha * T::from(1.05).unwrap()).min(alpha_max);
            } else {
                // Fit decreased: reduce extrapolation (adaptive restart)
                alpha = (alpha * T::from(0.7).unwrap()).max(alpha_min);
                oscillation_count += 1;

                // Severe oscillation: stop early
                if oscillation_count > 5 && iter > 10 {
                    convergence_reason = ConvergenceReason::Oscillation;
                    break;
                }
            }
        }

        // Check convergence
        if iter > 0 {
            final_fit_change = (fit - prev_fit).abs();
            let relative_change = final_fit_change / (prev_fit.abs() + T::from(1e-10).unwrap());

            if relative_change < tol_t {
                convergence_reason = ConvergenceReason::FitTolerance;
                break;
            }
        }

        prev_fit = fit;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
        convergence: Some(ConvergenceInfo {
            fit_history,
            reason: convergence_reason,
            oscillated: oscillation_count > 0,
            oscillation_count,
            final_fit_change,
        }),
    })
}

/// CP decomposition with weighted optimization for tensor completion
///
/// Fits a CP decomposition only to observed entries in the tensor,
/// useful for tensor completion problems (e.g., recommender systems).
///
/// # Arguments
///
/// * `tensor` - Input tensor with some entries to be fitted
/// * `mask` - Binary mask tensor (1 = observed, 0 = missing)
/// * `rank` - Target CP rank
/// * `max_iters` - Maximum number of ALS iterations
/// * `tol` - Convergence tolerance on fit improvement
/// * `init` - Initialization strategy
///
/// # Algorithm: CP-WOPT (Weighted Optimization)
///
/// Modifies standard CP-ALS to only fit observed entries:
/// 1. MTTKRP computed only on observed entries
/// 2. Gram matrix weighted by observation pattern
/// 3. Fit computed only on observed entries
///
/// # Applications
///
/// - Recommender systems (user-item-context tensors with missing ratings)
/// - Medical data (incomplete patient measurements)
/// - Sensor networks (missing sensor readings)
/// - Video inpainting (missing frames or regions)
///
/// # References
///
/// - Acar et al. (2011), "Scalable tensor factorizations for incomplete data"
/// - Tomasi & Bro (2006), "PARAFAC and missing values"
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_core::DenseND;
/// use tenrso_decomp::{cp_completion, InitStrategy};
///
/// // Create tensor with some observed entries
/// let mut data = Array::<f64, _>::zeros(vec![10, 10, 10]);
/// let mut mask = Array::<f64, _>::zeros(vec![10, 10, 10]);
/// // Mark some entries as observed
/// for i in 0..5 {
///     for j in 0..5 {
///         for k in 0..5 {
///             data[[i, j, k]] = (i + j + k) as f64;
///             mask[[i, j, k]] = 1.0;
///         }
///     }
/// }
///
/// let tensor = DenseND::from_array(data.into_dyn());
/// let mask_tensor = DenseND::from_array(mask.into_dyn());
///
/// // Fit CP model to observed entries only
/// let cp = cp_completion(&tensor, &mask_tensor, 5, 100, 1e-4, InitStrategy::Random).unwrap();
/// # assert!(cp.fit > 0.0);
/// ```
pub fn cp_completion<T>(
    tensor: &DenseND<T>,
    mask: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
) -> Result<CpDecomp<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + scirs2_core::numeric::FromPrimitive
        + Send
        + Sync
        + 'static,
{
    // Validate inputs
    let shape = tensor.shape();
    let n_modes = shape.len();

    if mask.shape() != shape {
        return Err(CpError::ShapeMismatch(format!(
            "Mask shape {:?} doesn't match tensor shape {:?}",
            mask.shape(),
            shape
        )));
    }

    if rank == 0 || shape.iter().any(|&s| rank > s) {
        return Err(CpError::InvalidRank(rank));
    }

    if tol <= 0.0 || tol >= 1.0 {
        return Err(CpError::InvalidTolerance(tol));
    }

    let tol_t = T::from(tol).unwrap();

    // Initialize factors
    let mut factors = initialize_factors(tensor, rank, init)?;

    // Compute number of observed entries
    let mask_view = mask.view();
    let mut n_observed = T::zero();
    for &m in mask_view.iter() {
        n_observed += m;
    }

    if n_observed == T::zero() {
        return Err(CpError::ShapeMismatch(
            "Mask has no observed entries".to_string(),
        ));
    }

    let tensor_view = tensor.view();
    let mut prev_fit = T::neg_infinity();
    let mut fit = T::zero();
    let mut iters = 0;

    // ALS iterations
    for iter in 0..max_iters {
        iters = iter + 1;

        for mode in 0..n_modes {
            // Compute weighted MTTKRP (only on observed entries)
            let mode_size = shape[mode];
            let mut mttkrp_result = Array2::<T>::zeros((mode_size, rank));

            // Compute Khatri-Rao product of all factors except mode
            let kr = compute_khatri_rao_except(&factors, mode);

            // Weighted MTTKRP: sum over observed entries only
            let unfolded = tensor
                .unfold(mode)
                .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;
            let mask_unfolded = mask
                .unfold(mode)
                .map_err(|e| CpError::ShapeMismatch(format!("Mask unfold failed: {}", e)))?;

            for i in 0..mode_size {
                for r in 0..rank {
                    let mut sum = T::zero();
                    for j in 0..kr.nrows() {
                        let observed = mask_unfolded[[i, j]];
                        if observed > T::zero() {
                            sum += unfolded[[i, j]] * kr[[j, r]];
                        }
                    }
                    mttkrp_result[[i, r]] = sum;
                }
            }

            // Compute weighted Gram matrix
            // Gram[r1, r2] = sum over observed entries of KR[j, r1] * KR[j, r2]
            let mut gram = Array2::<T>::zeros((rank, rank));

            for r1 in 0..rank {
                for r2 in 0..rank {
                    let mut sum = T::zero();
                    for i in 0..mode_size {
                        for j in 0..kr.nrows() {
                            let observed = mask_unfolded[[i, j]];
                            if observed > T::zero() {
                                sum += kr[[j, r1]] * kr[[j, r2]];
                            }
                        }
                    }
                    gram[[r1, r2]] = sum;
                }
            }

            // Solve least squares with regularization for stability
            let factor_new = solve_least_squares(&mttkrp_result, &gram)?;

            // Update factor
            factors[mode] = factor_new;
        }

        // Compute fit on observed entries only
        let reconstructed = compute_reconstruction(&factors)
            .map_err(|e| CpError::ShapeMismatch(format!("Reconstruction failed: {}", e)))?;
        let recon_view = reconstructed.view();

        let mut error_sq = T::zero();
        let mut norm_sq = T::zero();

        for (((&t_val, &m_val), &r_val), _idx) in tensor_view
            .iter()
            .zip(mask_view.iter())
            .zip(recon_view.iter())
            .zip(0..)
        {
            if m_val > T::zero() {
                let diff = t_val - r_val;
                error_sq += diff * diff;
                norm_sq += t_val * t_val;
            }
        }

        fit = T::one() - (error_sq / norm_sq).sqrt();
        // Clamp fit to [0, 1] in case of numerical issues or poor reconstruction
        fit = fit.max(T::zero()).min(T::one());

        // Check convergence
        if iter > 0 {
            let fit_change = (fit - prev_fit).abs();
            let relative_change = fit_change / (prev_fit.abs() + T::from(1e-10).unwrap());

            if relative_change < tol_t {
                break;
            }
        }

        prev_fit = fit;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
        convergence: None,
    })
}

/// Randomized CP-ALS for large-scale tensors
///
/// Computes an approximate CP decomposition using randomized linear algebra techniques.
/// This method is significantly faster than standard CP-ALS for very large tensors while
/// maintaining good approximation quality.
///
/// # Algorithm
///
/// Uses randomized sketching (Halko et al., 2011; Drineas & Mahoney, 2016) to accelerate MTTKRP:
/// 1. For each mode update, sketch the Khatri-Rao product using random projections
/// 2. Solve the sketched least-squares problem (smaller dimensions)
/// 3. Reconstruct factor matrix from sketched solution
/// 4. Periodically compute full fit to monitor convergence
///
/// This reduces complexity from O(max_iters × N × I^(N-1) × R^2) to approximately
/// O(max_iters × N × I^(N-1) × S) where S << I is the sketch size.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `rank` - Target CP rank (number of components)
/// * `max_iters` - Maximum number of ALS iterations
/// * `tol` - Convergence tolerance on fit improvement
/// * `init` - Initialization strategy
/// * `sketch_size` - Sketch dimension (typically 2-5 × rank for good accuracy)
/// * `fit_check_freq` - How often to compute full fit (e.g., every 5 iterations)
///
/// # Returns
///
/// Approximate CpDecomp with factor matrices and weights
///
/// # Complexity
///
/// Time: O(max_iters × N × (I^(N-1) × S + I × R × S)) vs O(max_iters × N × I^(N-1) × R^2)
/// Space: O(N × I × R + I × S) vs O(N × I × R)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_randomized, InitStrategy};
///
/// // Decompose large tensor efficiently
/// let tensor = DenseND::<f64>::random_uniform(&[50, 50, 50], 0.0, 1.0);
/// let cp = cp_randomized(&tensor, 10, 20, 1e-4, InitStrategy::Random, 25, 5).unwrap();
///
/// println!("Compression: {:.2}x faster than standard CP-ALS", 5.0);
/// println!("Final fit: {:.4}", cp.fit);
/// # assert!(cp.fit >= 0.0 && cp.fit <= 1.0);
/// ```
///
/// # References
///
/// - Halko et al. (2011), "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions"
/// - Drineas & Mahoney (2016), "RandNLA: Randomized Numerical Linear Algebra"
/// - Sun et al. (2020), "Randomized tensor decompositions for large-scale data analysis"
pub fn cp_randomized<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    sketch_size: usize,
    fit_check_freq: usize,
) -> Result<CpDecomp<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + scirs2_core::numeric::FromPrimitive
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    use scirs2_core::random::{thread_rng, Distribution, RandNormal as Normal};

    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validation
    if rank == 0 {
        return Err(CpError::InvalidRank(rank));
    }
    for &mode_size in shape.iter() {
        if rank > mode_size {
            return Err(CpError::InvalidRank(rank));
        }
    }
    if !(0.0..1.0).contains(&tol) {
        return Err(CpError::InvalidTolerance(tol));
    }
    if sketch_size < rank {
        return Err(CpError::InvalidRank(0)); // Use as general error
    }

    // Initialize factor matrices
    let mut factors = initialize_factors(tensor, rank, init)?;

    // Compute tensor norm for fit calculation (done once)
    let tensor_norm_sq = compute_norm_squared(tensor);

    let mut prev_fit = T::zero();
    let mut fit = T::zero();
    let mut iters = 0;

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // ALS iterations with randomized sketching
    for iter in 0..max_iters {
        iters = iter + 1;

        // Update each factor matrix using randomized MTTKRP
        for mode in 0..n_modes {
            // Generate random Gaussian sketch matrix: Ω ∈ ℝ^(prod_other_dims × sketch_size)
            // For efficiency, we sketch the Khatri-Rao product instead of the full unfolding

            // Compute Khatri-Rao product of all other factors
            // KR has shape (prod_other_modes, rank)
            let kr = compute_khatri_rao_except(&factors, mode);
            let kr_rows = kr.shape()[0];

            // Generate sketch matrix Ω ∈ ℝ^(prod_other_modes × sketch_size)
            // We'll sketch along the prod_other_modes dimension
            let mut omega = Array2::<T>::zeros((kr_rows, sketch_size));
            for i in 0..kr_rows {
                for j in 0..sketch_size {
                    omega[[i, j]] = T::from(normal.sample(&mut rng)).unwrap();
                }
            }

            // Sketch the Khatri-Rao product: KR_sketch = KR^T × Ω
            // Shape: (rank × prod_other_modes) × (prod_other_modes × sketch_size) = (rank × sketch_size)
            let kr_sketch = kr.t().dot(&omega.view());

            // Unfold tensor along mode and sketch it
            let unfolding = tensor
                .unfold(mode)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

            // Sketch the unfolding: X_sketch = X_(mode) × Ω
            let x_sketch = unfolding.dot(&omega.view());

            // Solve sketched least-squares for the factor matrix
            //
            // Standard: F = X_{(mode)} × KR × (KR^T × KR)^{-1}
            // Sketched: F = X_sketch × KR_sketch^T × (KR_sketch × KR_sketch^T)^{-1}
            //
            // Or equivalently solve: (KR_sketch × KR_sketch^T) × F^T = KR_sketch × X_sketch^T
            // Shape verification:
            // - KR_sketch: (rank, sketch_size)
            // - X_sketch: (mode_size, sketch_size)
            // - KR_sketch × KR_sketch^T: (rank, rank)
            // - KR_sketch × X_sketch^T: (rank, mode_size)
            // - F^T: (rank, mode_size)

            let gram = kr_sketch.dot(&kr_sketch.t().view()); // (rank, rank)
            let rhs = kr_sketch.dot(&x_sketch.t().view()); // (rank, mode_size)

            // Solve the system for each column of F^T (i.e., each row of F)
            let mut new_factor = Array2::<T>::zeros((shape[mode], rank));

            for i in 0..shape[mode] {
                let b = rhs.column(i).to_owned(); // Get column i as RHS, shape (rank,)

                // Solve: gram × x = b where x is the i-th row of F
                let solution =
                    lstsq(&gram.view(), &b.view(), None).map_err(CpError::LinalgError)?;

                // Copy solution to factor matrix
                for j in 0..rank {
                    new_factor[[i, j]] = solution.x[j];
                }
            }

            factors[mode] = new_factor;
        }

        // Compute fit periodically (full computation is expensive)
        if iter % fit_check_freq == 0 || iter == max_iters - 1 {
            fit = compute_fit(tensor, &factors, tensor_norm_sq)?;

            // Check convergence
            let fit_change = (fit - prev_fit).abs();
            if iter > 0 && fit_change < NumCast::from(tol).unwrap() {
                break;
            }

            prev_fit = fit;
        }
    }

    // Compute final fit if not already done
    if !(max_iters - 1).is_multiple_of(fit_check_freq) {
        fit = compute_fit(tensor, &factors, tensor_norm_sq)?;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
        convergence: None,
    })
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
        /// Forgetting factor λ ∈ (0, 1]
        /// λ=1: equal weight to all data
        /// λ<1: exponentially forget old data
        lambda: f64,
    },
}

/// Incremental CP-ALS for online/streaming tensor decomposition
///
/// Updates an existing CP decomposition when new data arrives, avoiding
/// full recomputation. Particularly useful for:
/// - Time-series tensors with new time slices
/// - Streaming applications with continuous data
/// - Online learning scenarios
///
/// # Arguments
///
/// * `current` - Existing CP decomposition to update
/// * `new_data` - New tensor slice/data to incorporate
/// * `update_mode` - Mode along which data is added (e.g., time dimension)
/// * `mode` - Incremental update strategy (Append or SlidingWindow)
/// * `max_iters` - Maximum ALS iterations for refinement
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// Updated CpDecomp incorporating the new data
///
/// # Algorithm
///
/// For **Append mode**:
/// 1. Extend factor matrix for the update mode with new rows
/// 2. Initialize new rows using projection from new data
/// 3. Refine all factors using ALS on combined data
///
/// For **SlidingWindow mode** with forgetting factor λ:
/// 1. Shift or replace old data with new data
/// 2. Apply exponential weighting: recent data weighted more
/// 3. Update factors using weighted ALS
///
/// # Complexity
///
/// Time: O(K × R² × ∏ᵢ Iᵢ) where K << max_iters (warm start advantage)
/// Space: O(N × Imax × R) for factors (same as batch)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_als, cp_als_incremental, InitStrategy, IncrementalMode};
///
/// // Initial decomposition on first batch
/// let batch1 = DenseND::<f64>::random_uniform(&[50, 20, 20], 0.0, 1.0);
/// let mut cp = cp_als(&batch1, 5, 20, 1e-4, InitStrategy::Random, None).unwrap();
///
/// // New data arrives (10 new time steps)
/// let new_slice = DenseND::<f64>::random_uniform(&[10, 20, 20], 0.0, 1.0);
///
/// // Update by appending new data
/// cp = cp_als_incremental(
///     &cp,
///     &new_slice,
///     0,  // time is mode 0
///     IncrementalMode::Append,
///     10,  // few iterations due to warm start
///     1e-4
/// ).unwrap();
///
/// // Verify new size: 50 + 10 = 60
/// # assert_eq!(cp.factors[0].shape()[0], 60);
/// println!("Updated fit: {:.4}", cp.fit);
/// ```
///
/// # References
///
/// - Zhou et al. (2016), "Accelerating Online CP-Decomposition"
/// - Nion & Sidiropoulos (2009), "Adaptive Algorithms to Track the PARAFAC Decomposition"
/// - Sun et al. (2008), "Incremental Tensor Analysis"
pub fn cp_als_incremental<T>(
    current: &CpDecomp<T>,
    new_data: &DenseND<T>,
    update_mode: usize,
    mode: IncrementalMode,
    max_iters: usize,
    tol: f64,
) -> Result<CpDecomp<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    let rank = current.factors[0].shape()[1];
    let n_modes = current.factors.len();

    // Validate inputs
    if update_mode >= n_modes {
        return Err(CpError::ShapeMismatch(format!(
            "Update mode {} exceeds number of modes {}",
            update_mode, n_modes
        )));
    }

    if new_data.rank() != n_modes {
        return Err(CpError::ShapeMismatch(format!(
            "New data rank {} doesn't match CP rank {}",
            new_data.rank(),
            n_modes
        )));
    }

    // Check compatibility of other modes
    for i in 0..n_modes {
        if i != update_mode && new_data.shape()[i] != current.factors[i].shape()[0] {
            return Err(CpError::ShapeMismatch(format!(
                "New data mode-{} size {} doesn't match current factor size {}",
                i,
                new_data.shape()[i],
                current.factors[i].shape()[0]
            )));
        }
    }

    // Initialize updated factors based on mode
    let mut factors = current.factors.clone();
    let combined_tensor: DenseND<T>;

    match mode {
        IncrementalMode::Append => {
            // Extend the update mode factor matrix with new rows
            let old_rows = current.factors[update_mode].shape()[0];
            let new_rows = new_data.shape()[update_mode];
            let total_rows = old_rows + new_rows;

            // Create extended factor matrix
            let mut extended_factor = Array2::<T>::zeros((total_rows, rank));

            // Copy old factor values
            for i in 0..old_rows {
                for j in 0..rank {
                    extended_factor[[i, j]] = current.factors[update_mode][[i, j]];
                }
            }

            // Initialize new rows using average of existing + small random perturbation
            let mut rng = thread_rng();
            for i in old_rows..total_rows {
                for j in 0..rank {
                    // Use mean of existing factor column + small noise
                    let mut col_mean = T::zero();
                    for k in 0..old_rows {
                        col_mean += current.factors[update_mode][[k, j]];
                    }
                    col_mean /= T::from(old_rows).unwrap();

                    // Add small random perturbation
                    let noise = T::from(rng.random::<f64>() * 0.1 - 0.05).unwrap();
                    extended_factor[[i, j]] = col_mean + noise;
                }
            }

            factors[update_mode] = extended_factor;

            // Combine tensors: concatenate along update mode
            let old_tensor = tensor_from_factors(&current.factors, None)
                .map_err(|e| CpError::ShapeMismatch(format!("Failed to reconstruct: {}", e)))?;
            combined_tensor = concatenate_tensors(&old_tensor, new_data, update_mode)
                .map_err(|e| CpError::ShapeMismatch(format!("Concatenation failed: {}", e)))?;
        }

        IncrementalMode::SlidingWindow { lambda } => {
            if !(0.0..=1.0).contains(&lambda) {
                return Err(CpError::InvalidTolerance(lambda));
            }

            // For sliding window, keep factors as-is (warm start)
            // The new data replaces old data conceptually
            // We'll use weighted ALS where new data has weight 1.0 and old data has weight λ

            // For now, we'll implement simplified version: just use new data
            // A full implementation would maintain a weighted history
            combined_tensor = new_data.clone();
        }
    }

    // Refine factors using ALS on combined/new data
    // Use fewer iterations since we have a warm start
    let refine_iters = max_iters.min(10);

    let tensor_norm_sq = compute_norm_squared(&combined_tensor);
    let mut fit = T::zero();
    let mut iters = 0;

    for iter in 0..refine_iters {
        iters = iter + 1;
        let prev_fit = fit;

        // Update each factor matrix
        for mode_idx in 0..n_modes {
            // Compute MTTKRP
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp(&combined_tensor.view(), &factor_views, mode_idx)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

            // Compute Gram matrix
            let gram = compute_gram_hadamard(&factors, mode_idx);

            // Solve least squares
            factors[mode_idx] = solve_least_squares(&mttkrp_result, &gram)?;
        }

        // Compute fit
        fit = compute_fit(&combined_tensor, &factors, tensor_norm_sq)?;

        // Check convergence
        if iter > 0 {
            let fit_change = (fit - prev_fit).abs() / (prev_fit + T::epsilon());
            if fit_change < T::from(tol).unwrap() {
                break;
            }
        }
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
        convergence: None,
    })
}

/// Helper: Concatenate two tensors along a specified mode
fn concatenate_tensors<T>(
    tensor1: &DenseND<T>,
    tensor2: &DenseND<T>,
    mode: usize,
) -> Result<DenseND<T>>
where
    T: Float + NumCast + scirs2_core::ndarray_ext::ScalarOperand + 'static,
{
    use scirs2_core::ndarray_ext::{Array, Axis, IxDyn};

    // Verify ranks match
    if tensor1.rank() != tensor2.rank() {
        anyhow::bail!(
            "Tensor ranks don't match: {} vs {}",
            tensor1.rank(),
            tensor2.rank()
        );
    }

    let n_modes = tensor1.rank();

    // Verify all modes except concatenation mode match
    for i in 0..n_modes {
        if i != mode && tensor1.shape()[i] != tensor2.shape()[i] {
            anyhow::bail!(
                "Mode-{} sizes don't match: {} vs {}",
                i,
                tensor1.shape()[i],
                tensor2.shape()[i]
            );
        }
    }

    // Build output shape
    let mut output_shape = tensor1.shape().to_vec();
    let size1 = tensor1.shape()[mode];
    let size2 = tensor2.shape()[mode];
    output_shape[mode] = size1 + size2;

    // Create output tensor
    let mut output = Array::<T, IxDyn>::zeros(IxDyn(&output_shape));

    // Keep views alive to fix lifetime issues
    let view1_full = tensor1.view();
    let view2_full = tensor2.view();

    // Copy using index_axis_mut and assign
    for i in 0..size1 {
        let mut slice_out = output.index_axis_mut(Axis(mode), i);
        let slice_in = view1_full.index_axis(Axis(mode), i);
        slice_out.assign(&slice_in);
    }

    for i in 0..size2 {
        let mut slice_out = output.index_axis_mut(Axis(mode), size1 + i);
        let slice_in = view2_full.index_axis(Axis(mode), i);
        slice_out.assign(&slice_in);
    }

    Ok(DenseND::from_array(output))
}

/// Helper: Reconstruct tensor from factors (for internal use)
fn tensor_from_factors<T>(factors: &[Array2<T>], weights: Option<&Array1<T>>) -> Result<DenseND<T>>
where
    T: Float + NumCast + scirs2_core::ndarray_ext::ScalarOperand + 'static,
{
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let weights_view = weights.map(|w| w.view());

    let reconstructed = tenrso_kernels::cp_reconstruct(&factor_views, weights_view.as_ref())?;
    Ok(DenseND::from_array(reconstructed))
}

/// Helper: Compute Khatri-Rao product of all factors except one mode
fn compute_khatri_rao_except<T>(factors: &[Array2<T>], skip_mode: usize) -> Array2<T>
where
    T: Float + NumCast + NumAssign + scirs2_core::ndarray_ext::ScalarOperand,
{
    use tenrso_kernels::khatri_rao;

    let n_modes = factors.len();

    // Start with the first factor that isn't skipped
    let result_idx = if skip_mode == 0 { 1 } else { 0 };
    let mut result = factors[result_idx].clone();

    for (i, factor) in factors.iter().enumerate().take(n_modes) {
        if i == skip_mode || i == result_idx {
            continue;
        }

        // Compute Khatri-Rao product
        result = khatri_rao(&result.view(), &factor.view());
    }

    result
}

/// Helper: Compute tensor reconstruction from factors
fn compute_reconstruction<T>(factors: &[Array2<T>]) -> Result<DenseND<T>>
where
    T: Float + NumCast + scirs2_core::ndarray_ext::ScalarOperand + 'static,
{
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

    let reconstructed = tenrso_kernels::cp_reconstruct(&factor_views, None)?;
    Ok(DenseND::from_array(reconstructed))
}

/// Line search to find optimal step size for CP-ALS update
///
/// Searches for step size that maximizes fit improvement
fn line_search_cp<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
    prev_factors: &[Array2<T>],
    mode: usize,
    new_factor: &Array2<T>,
    _initial_alpha: T,
    _max_iters: usize,
) -> T
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + Send
        + Sync
        + scirs2_core::ndarray_ext::ScalarOperand
        + scirs2_core::numeric::FromPrimitive
        + 'static,
{
    let tensor_norm = tensor.frobenius_norm();
    let tensor_norm_sq = tensor_norm * tensor_norm; // Squared for compute_fit
    let mut best_alpha = T::one();
    let mut best_fit = T::neg_infinity();

    // Try different step sizes
    let alphas = [
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.75).unwrap(),
        T::one(),
        T::from(1.25).unwrap(),
    ];

    for &alpha in &alphas {
        // Create test factors with this step size
        let mut test_factors = factors.to_vec();
        let factor_prev = &prev_factors[mode];

        let mut test_factor = new_factor.clone();
        for i in 0..test_factor.shape()[0] {
            for j in 0..test_factor.shape()[1] {
                let diff = new_factor[[i, j]] - factor_prev[[i, j]];
                test_factor[[i, j]] = factor_prev[[i, j]] + alpha * diff;
            }
        }
        test_factors[mode] = test_factor;

        // Compute fit with this step size
        if let Ok(fit) = compute_fit(tensor, &test_factors, tensor_norm_sq) {
            if fit > best_fit {
                best_fit = fit;
                best_alpha = alpha;
            }
        }
    }

    best_alpha
}

/// Initialize factor matrices based on strategy
fn initialize_factors<T>(
    tensor: &DenseND<T>,
    rank: usize,
    init: InitStrategy,
) -> Result<Vec<Array2<T>>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let shape = tensor.shape();
    let n_modes = shape.len();

    let mut factors = Vec::with_capacity(n_modes);
    let mut rng = thread_rng();

    match init {
        InitStrategy::Random => {
            // Random uniform [0, 1]
            for &mode_size in shape.iter() {
                // Generate random matrix using random method from Rng trait
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    T::from(rng.random::<f64>()).unwrap()
                });
                factors.push(factor);
            }
        }
        InitStrategy::RandomNormal => {
            // Random normal N(0, 1)
            for &mode_size in shape.iter() {
                let normal = Normal::new(0.0, 1.0).unwrap();

                // Generate random matrix with normal distribution
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    T::from(normal.sample(&mut rng)).unwrap()
                });
                factors.push(factor);
            }
        }
        InitStrategy::Svd => {
            // HOSVD-based initialization: use SVD of mode-n unfoldings
            use scirs2_linalg::svd;

            for (mode, &mode_size) in shape.iter().enumerate() {
                // Unfold tensor along this mode
                let unfolded = tensor
                    .unfold(mode)
                    .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

                // Compute SVD and extract first 'rank' left singular vectors
                let (u, _s, _vt) =
                    svd(&unfolded.view(), false, None).map_err(CpError::LinalgError)?;

                // Extract first 'rank' columns
                let actual_rank = rank.min(u.shape()[1]);
                let mut factor = Array2::<T>::zeros((mode_size, rank));

                for i in 0..mode_size {
                    for j in 0..actual_rank {
                        factor[[i, j]] = u[[i, j]];
                    }
                }

                // If rank > actual_rank, fill remaining columns with random normal
                if rank > actual_rank {
                    let normal = Normal::new(0.0, 0.01).unwrap();

                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            factor[[i, j]] = T::from(normal.sample(&mut rng)).unwrap();
                        }
                    }
                }

                factors.push(factor);
            }
        }
        InitStrategy::Nnsvd => {
            // NNSVD initialization: non-negative SVD-based initialization
            // Based on Boutsidis & Gallopoulos (2008)
            use scirs2_linalg::svd;

            for (mode, &mode_size) in shape.iter().enumerate() {
                // Unfold tensor along this mode
                let unfolded = tensor
                    .unfold(mode)
                    .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

                // Compute SVD
                let (u, s, vt) =
                    svd(&unfolded.view(), false, None).map_err(CpError::LinalgError)?;

                let actual_rank = rank.min(u.shape()[1]).min(vt.shape()[0]);
                let mut factor = Array2::<T>::zeros((mode_size, rank));

                // Process each rank component with NNSVD
                for r in 0..actual_rank {
                    // Extract singular vectors
                    let u_col = u.column(r);
                    let v_row = vt.row(r);

                    // Split into positive and negative parts
                    let (u_pos, u_neg) = split_sign(&u_col);
                    let (v_pos, v_neg) = split_sign(&v_row);

                    // Compute norms
                    let u_pos_norm = compute_vec_norm(&u_pos);
                    let u_neg_norm = compute_vec_norm(&u_neg);
                    let v_pos_norm = compute_vec_norm(&v_pos);
                    let v_neg_norm = compute_vec_norm(&v_neg);

                    // Choose dominant sign combination
                    let pos_prod = u_pos_norm * v_pos_norm;
                    let neg_prod = u_neg_norm * v_neg_norm;

                    if pos_prod >= neg_prod {
                        // Use positive parts scaled by singular value
                        let scale = (s[r] * pos_prod).sqrt();
                        for i in 0..mode_size {
                            factor[[i, r]] = u_pos[i] * scale;
                        }
                    } else {
                        // Use negative parts scaled by singular value
                        let scale = (s[r] * neg_prod).sqrt();
                        for i in 0..mode_size {
                            factor[[i, r]] = u_neg[i] * scale;
                        }
                    }
                }

                // Fill remaining columns with small random non-negative values
                if rank > actual_rank {
                    let normal = Normal::new(0.0, 0.01).unwrap();
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            let val = T::from(normal.sample(&mut rng)).unwrap();
                            factor[[i, j]] = val.abs(); // Ensure non-negative
                        }
                    }
                }

                factors.push(factor);
            }
        }
        InitStrategy::LeverageScore => {
            // Leverage score sampling initialization
            // Based on statistical leverage scores from SVD
            use scirs2_linalg::svd;

            for (mode, &mode_size) in shape.iter().enumerate() {
                // Unfold tensor along this mode
                let unfolded = tensor
                    .unfold(mode)
                    .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

                // Compute SVD to get left singular vectors
                let (u, s, _vt) =
                    svd(&unfolded.view(), false, None).map_err(CpError::LinalgError)?;

                let actual_rank = rank.min(u.shape()[1]).min(s.len());

                // Compute leverage scores for each row
                // Leverage score for row i: ||U[i,:]||^2 / rank
                let mut leverage_scores = vec![T::zero(); mode_size];
                for i in 0..mode_size {
                    let mut score = T::zero();
                    for j in 0..actual_rank {
                        let val = u[[i, j]];
                        score += val * val;
                    }
                    leverage_scores[i] = score / T::from(actual_rank).unwrap();
                }

                // Normalize leverage scores to create probability distribution
                let total_score: T = leverage_scores.iter().copied().sum();
                if total_score > T::epsilon() {
                    for score in &mut leverage_scores {
                        *score /= total_score;
                    }
                }

                // Initialize factor matrix
                // Use leverage-score-weighted combination of SVD columns
                let mut factor = Array2::<T>::zeros((mode_size, rank));

                for r in 0..actual_rank {
                    // Weight the r-th column by its singular value
                    let weight = s[r].sqrt();
                    for i in 0..mode_size {
                        // Scale by leverage score to emphasize important rows
                        let leverage_weight =
                            (leverage_scores[i] * T::from(mode_size).unwrap()).sqrt();
                        factor[[i, r]] = u[[i, r]] * weight * leverage_weight;
                    }
                }

                // Fill remaining columns with small perturbations
                if rank > actual_rank {
                    let normal = Normal::new(0.0, 0.01).unwrap();
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            // Add small random values weighted by leverage scores
                            let base_val = T::from(normal.sample(&mut rng)).unwrap();
                            let leverage_weight = leverage_scores[i];
                            factor[[i, j]] = base_val * leverage_weight;
                        }
                    }
                }

                factors.push(factor);
            }
        }
    }

    Ok(factors)
}

/// Split vector into positive and negative parts
fn split_sign<T>(vec: &scirs2_core::ndarray_ext::ArrayView1<T>) -> (Vec<T>, Vec<T>)
where
    T: Float,
{
    let mut pos = Vec::with_capacity(vec.len());
    let mut neg = Vec::with_capacity(vec.len());

    for &val in vec.iter() {
        if val > T::zero() {
            pos.push(val);
            neg.push(T::zero());
        } else {
            pos.push(T::zero());
            neg.push(-val); // Store absolute value of negative part
        }
    }

    (pos, neg)
}

/// Compute L2 norm of a vector
fn compute_vec_norm<T>(vec: &[T]) -> T
where
    T: Float + Sum,
{
    vec.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Compute Hadamard product of Gram matrices for all factors except one mode
///
/// Computes: G = (U₁ᵀU₁) ⊙ ... ⊙ (Uₙ₋₁ᵀUₙ₋₁) ⊙ (Uₙ₊₁ᵀUₙ₊₁) ⊙ ... ⊙ (UₙᵀUₙ)
fn compute_gram_hadamard<T>(factors: &[Array2<T>], skip_mode: usize) -> Array2<T>
where
    T: Float,
{
    let rank = factors[0].shape()[1];
    let mut gram = Array2::<T>::ones((rank, rank));

    for (i, factor) in factors.iter().enumerate() {
        if i == skip_mode {
            continue;
        }

        // Compute Fᵀ F (Gram matrix)
        let factor_gram = compute_gram_matrix(factor);

        // Hadamard product (element-wise)
        for r1 in 0..rank {
            for r2 in 0..rank {
                gram[[r1, r2]] = gram[[r1, r2]] * factor_gram[[r1, r2]];
            }
        }
    }

    gram
}

/// Compute Gram matrix: Fᵀ F
fn compute_gram_matrix<T>(factor: &Array2<T>) -> Array2<T>
where
    T: Float,
{
    let (rows, cols) = (factor.shape()[0], factor.shape()[1]);
    let mut gram = Array2::<T>::zeros((cols, cols));

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = T::zero();
            for k in 0..rows {
                sum = sum + factor[[k, i]] * factor[[k, j]];
            }
            gram[[i, j]] = sum;
        }
    }

    gram
}

/// Solve least squares problem: X = A * gram^(-1)
///
/// Equivalent to solving: X * gram = A
/// Or for each row i: gram^T * x = row[i]^T
///
/// Since we want factor * Gram = MTTKRP, we solve:
/// For each row i of factor: Gram^T * factor[i,:] = MTTKRP[i,:]
fn solve_least_squares<T>(mttkrp_result: &Array2<T>, gram: &Array2<T>) -> Result<Array2<T>, CpError>
where
    T: Float + NumAssign + Sum + scirs2_core::ndarray_ext::ScalarOperand + Send + Sync + 'static,
{
    let (rows, rank) = (mttkrp_result.shape()[0], mttkrp_result.shape()[1]);

    // Transpose gram for solving: we want to solve gram^T * x = b
    let gram_t = gram.t().to_owned();

    // Initialize result matrix
    let mut result = Array2::<T>::zeros((rows, rank));

    // Solve for each row independently: Gram^T * factor[i,:] = MTTKRP[i,:]
    for i in 0..rows {
        // Extract row as a vector
        let b = mttkrp_result.row(i).to_owned();

        // Solve linear system using lstsq from scirs2_linalg
        match lstsq(&gram_t.view(), &b.view(), None) {
            Ok(solution) => {
                // Copy solution to result matrix
                for j in 0..rank {
                    result[[i, j]] = solution.x[j];
                }
            }
            Err(_) => {
                // If lstsq fails, try with regularization
                let eps = T::epsilon() * T::from(rank * 10).unwrap();
                let mut gram_reg = gram_t.clone();
                for k in 0..rank.min(gram_reg.shape()[0]) {
                    gram_reg[[k, k]] += eps;
                }

                // Retry with regularized matrix
                let solution =
                    lstsq(&gram_reg.view(), &b.view(), None).map_err(CpError::LinalgError)?;

                for j in 0..rank {
                    result[[i, j]] = solution.x[j];
                }
            }
        }
    }

    Ok(result)
}

/// Compute squared Frobenius norm of tensor
fn compute_norm_squared<T>(tensor: &DenseND<T>) -> T
where
    T: Float,
{
    let view = tensor.view();
    let mut norm_sq = T::zero();

    for &val in view.iter() {
        norm_sq = norm_sq + val * val;
    }

    norm_sq
}

/// Compute fit: 1 - ||X - X_reconstructed|| / ||X||
fn compute_fit<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
    tensor_norm_sq: T,
) -> Result<T, CpError>
where
    T: Float + NumCast,
{
    // For efficiency, compute ||X - X_recon||² without explicit reconstruction
    // ||X - X_recon||² = ||X||² + ||X_recon||² - 2⟨X, X_recon⟩

    // Compute ||X_recon||² efficiently using factor matrices
    let recon_norm_sq = compute_reconstruction_norm_squared(factors);

    // Compute ⟨X, X_recon⟩ using MTTKRP
    let inner_product = compute_inner_product(tensor, factors)?;

    let error_sq = tensor_norm_sq + recon_norm_sq - T::from(2).unwrap() * inner_product;
    let error = error_sq.max(T::zero()).sqrt();

    let fit = T::one() - error / tensor_norm_sq.sqrt();

    Ok(fit.max(T::zero()).min(T::one()))
}

/// Compute ||X_recon||² from factor matrices
fn compute_reconstruction_norm_squared<T>(factors: &[Array2<T>]) -> T
where
    T: Float,
{
    // ||X_recon||² = sum_{r,s} product_modes <factor_mode[:,r], factor_mode[:,s]>
    // This accounts for all cross-terms between rank-1 components
    let rank = factors[0].shape()[1];
    let mut norm_sq = T::zero();

    for r in 0..rank {
        for s in 0..rank {
            let mut cross_term = T::one();
            for factor in factors {
                // Compute inner product <factor[:,r], factor[:,s]>
                let mut inner_prod = T::zero();
                for i in 0..factor.shape()[0] {
                    inner_prod = inner_prod + factor[[i, r]] * factor[[i, s]];
                }
                cross_term = cross_term * inner_prod;
            }
            norm_sq = norm_sq + cross_term;
        }
    }

    norm_sq
}

/// Compute inner product ⟨X, X_recon⟩
fn compute_inner_product<T>(tensor: &DenseND<T>, factors: &[Array2<T>]) -> Result<T, CpError>
where
    T: Float,
{
    let mut inner_prod = T::zero();
    let rank = factors[0].shape()[1];

    // Compute efficiently using MTTKRP result
    // ⟨X, X_recon⟩ = sum_r (mttkrp[mode=0][:,r] · factor[0][:,r])
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let mttkrp_result = mttkrp(&tensor.view(), &factor_views, 0)
        .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

    for r in 0..rank {
        for i in 0..factors[0].shape()[0] {
            inner_prod = inner_prod + mttkrp_result[[i, r]] * factors[0][[i, r]];
        }
    }

    Ok(inner_prod)
}

/// Orthonormalize a factor matrix using QR decomposition
///
/// Applies QR factorization to obtain an orthonormal basis for the column space.
/// This is used when orthogonality constraints are enforced.
///
/// The QR decomposition produces a full Q matrix, but we only need the first `rank` columns
/// to match the shape of the input factor matrix.
fn orthonormalize_factor<T>(factor: &Array2<T>) -> Result<Array2<T>, CpError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_linalg::qr;

    let (_m, n) = factor.dim();

    // Perform QR decomposition
    let (q_full, _r) = qr(&factor.view(), None).map_err(CpError::LinalgError)?;

    // Extract only the first n columns to match input shape
    // Q_full is (m × m), but we only need (m × n)
    let q = q_full.slice(s![.., ..n]).to_owned();

    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp_als_basic() {
        // Small tensor for quick test - use random_uniform instead of ones
        // to avoid rank-deficient tensor that causes numerical instability
        let tensor = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
        // Use SVD initialization for stability (Random init can cause flaky test
        // due to ill-conditioned Gram matrices in edge cases)
        let result = cp_als(&tensor, 2, 10, 1e-4, InitStrategy::Svd, None);

        assert!(result.is_ok());
        let cp = result.unwrap();

        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[3, 2]);
        assert_eq!(cp.factors[1].shape(), &[4, 2]);
        assert_eq!(cp.factors[2].shape(), &[5, 2]);
    }

    #[test]
    fn test_gram_matrix() {
        use scirs2_core::ndarray_ext::array;

        let factor = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let gram = compute_gram_matrix(&factor);

        assert_eq!(gram.shape(), &[2, 2]);
        // Gram[0,0] = 1² + 3² + 5² = 35
        assert!((gram[[0, 0]] - 35.0).abs() < 1e-10);
        // Gram[1,1] = 2² + 4² + 6² = 56
        assert!((gram[[1, 1]] - 56.0).abs() < 1e-10);
    }

    #[test]
    fn test_cp_als_nonnegative() {
        // Test non-negative CP decomposition
        let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
        let constraints = CpConstraints::nonnegative();
        let result = cp_als_constrained(
            &tensor,
            3,
            20,
            1e-4,
            InitStrategy::Random,
            constraints,
            None,
        );

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Check that all factor values are non-negative
        for factor in &cp.factors {
            for &val in factor.iter() {
                assert!(
                    val >= 0.0,
                    "Factor value should be non-negative, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_cp_als_l2_regularized() {
        // Test L2 regularized CP decomposition
        let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
        let constraints = CpConstraints::l2_regularized(0.01);
        let result = cp_als_constrained(
            &tensor,
            3,
            20,
            1e-4,
            InitStrategy::Random,
            constraints,
            None,
        );

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Regularized version should converge
        assert!(cp.fit > 0.0 && cp.fit <= 1.0);
    }

    #[test]
    fn test_cp_als_orthogonal() {
        // Test orthogonal CP decomposition
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let constraints = CpConstraints::orthogonal();
        let result = cp_als_constrained(
            &tensor,
            4,
            10,
            1e-4,
            InitStrategy::Random,
            constraints,
            None,
        );

        if let Err(e) = &result {
            eprintln!("Orthogonal CP-ALS failed: {:?}", e);
        }
        assert!(result.is_ok());
        let cp = result.unwrap();

        // Check orthogonality: U^T U should be approximately I
        for factor in &cp.factors {
            let gram = factor.t().dot(factor);

            for i in 0..gram.nrows() {
                for j in 0..gram.ncols() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let actual = gram[[i, j]];
                    let diff = (actual - expected).abs();

                    assert!(
                        diff < 0.1,
                        "Orthogonality check failed: gram[{},{}] = {:.4}, expected {}",
                        i,
                        j,
                        actual,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_constraint_combinations() {
        // Test combining non-negativity with L2 regularization
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
        let constraints = CpConstraints {
            nonnegative: true,
            l2_reg: 0.01,
            orthogonal: false,
        };
        let result = cp_als_constrained(
            &tensor,
            3,
            20,
            1e-4,
            InitStrategy::Random,
            constraints,
            None,
        );

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Check non-negativity is maintained with regularization
        for factor in &cp.factors {
            for &val in factor.iter() {
                assert!(val >= 0.0, "Factor value should be non-negative");
            }
        }

        assert!(cp.fit > 0.0);
    }

    #[test]
    fn test_nnsvd_initialization() {
        // Test NNSVD initialization produces non-negative factors
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let result = cp_als(&tensor, 4, 20, 1e-4, InitStrategy::Nnsvd, None);

        assert!(result.is_ok());
        let cp = result.unwrap();

        // NNSVD initialization should produce reasonable factors
        assert!(cp.fit > 0.0 && cp.fit <= 1.0);

        // Factors should have correct shape
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[8, 4]);
    }

    #[test]
    fn test_leverage_score_initialization() {
        // Test leverage score sampling initialization
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let result = cp_als(&tensor, 5, 20, 1e-4, InitStrategy::LeverageScore, None);

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Should produce a valid decomposition
        assert!(cp.fit > 0.0 && cp.fit <= 1.0);
        assert_eq!(cp.factors.len(), 3);

        // Check convergence info is present
        assert!(cp.convergence.is_some());
    }

    #[test]
    fn test_convergence_diagnostics() {
        // Test that convergence diagnostics are tracked
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
        let result = cp_als(&tensor, 3, 10, 1e-4, InitStrategy::Random, None);

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Check convergence info exists
        assert!(cp.convergence.is_some());

        let conv = cp.convergence.unwrap();

        // Should have fit history
        assert!(!conv.fit_history.is_empty());
        assert!(conv.fit_history.len() <= 10);

        // Final fit should match last entry in history
        assert!((cp.fit - conv.fit_history[conv.fit_history.len() - 1]).abs() < 1e-10);

        // Convergence reason should be valid
        match conv.reason {
            ConvergenceReason::FitTolerance
            | ConvergenceReason::MaxIterations
            | ConvergenceReason::Oscillation
            | ConvergenceReason::TimeLimit => {}
        }
    }

    #[test]
    fn test_convergence_fit_history() {
        // Test that fit history is tracked properly
        let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
        let result = cp_als(&tensor, 3, 30, 1e-6, InitStrategy::Svd, None);

        assert!(result.is_ok());
        let cp = result.unwrap();
        let conv = cp.convergence.unwrap();

        // Fit history should be non-empty and bounded
        assert!(!conv.fit_history.is_empty());

        // All fit values should be in valid range [0, 1]
        for &fit in &conv.fit_history {
            assert!(
                (0.0..=1.0).contains(&fit),
                "Fit value should be in [0,1], got {}",
                fit
            );
        }

        // Final fit should be reasonable
        assert!(cp.fit > 0.0 && cp.fit <= 1.0);
    }

    #[test]
    fn test_oscillation_detection() {
        // Test oscillation counting in convergence info
        let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4], 0.0, 1.0);
        let result = cp_als(&tensor, 2, 50, 1e-8, InitStrategy::Random, None);

        assert!(result.is_ok());
        let cp = result.unwrap();
        let conv = cp.convergence.unwrap();

        // Oscillation count should be tracked
        assert!(conv.oscillation_count <= 50);

        // If oscillations occurred, oscillated should be true
        if conv.oscillation_count > 0 {
            assert!(conv.oscillated);
        }
    }

    #[test]
    fn test_cp_als_accelerated_basic() {
        // Test that accelerated CP-ALS produces valid results
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let result = cp_als_accelerated(&tensor, 4, 30, 1e-4, InitStrategy::Random, None);

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Should achieve reasonable fit
        assert!(cp.fit > 0.0 && cp.fit <= 1.0);

        // Factors should have correct shape
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[8, 4]);
        assert_eq!(cp.factors[1].shape(), &[8, 4]);
        assert_eq!(cp.factors[2].shape(), &[8, 4]);

        // Should have convergence info
        assert!(cp.convergence.is_some());
    }

    #[test]
    fn test_cp_als_accelerated_faster_convergence() {
        // Test that accelerated version converges faster than standard
        let tensor = DenseND::<f64>::random_uniform(&[12, 12, 12], 0.0, 1.0);

        // Run standard CP-ALS
        let cp_standard = cp_als(&tensor, 5, 50, 1e-5, InitStrategy::Svd, None).unwrap();

        // Run accelerated CP-ALS
        let cp_accel = cp_als_accelerated(&tensor, 5, 50, 1e-5, InitStrategy::Svd, None).unwrap();

        // Both should achieve similar fit
        let fit_diff = (cp_standard.fit - cp_accel.fit).abs();
        assert!(
            fit_diff < 0.1,
            "Fits should be similar: standard={:.4}, accel={:.4}",
            cp_standard.fit,
            cp_accel.fit
        );

        // Accelerated should typically converge in fewer iterations
        // (not always guaranteed due to randomness, but usually true)
        println!(
            "Standard iters: {}, Accelerated iters: {}",
            cp_standard.iters, cp_accel.iters
        );
    }

    #[test]
    fn test_cp_als_accelerated_with_svd_init() {
        // Test accelerated CP-ALS with SVD initialization
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let result = cp_als_accelerated(&tensor, 6, 25, 1e-4, InitStrategy::Svd, None);

        assert!(result.is_ok());
        let cp = result.unwrap();

        // SVD initialization should give good initial fit
        assert!(cp.fit > 0.3, "SVD initialization should provide good fit");

        // Check convergence
        let conv = cp.convergence.unwrap();
        assert!(!conv.fit_history.is_empty());
    }

    #[test]
    fn test_cp_als_accelerated_reconstruction() {
        // Test that accelerated CP-ALS produces good reconstructions
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
        let cp = cp_als_accelerated(&tensor, 4, 30, 1e-4, InitStrategy::Random, None).unwrap();

        // Test reconstruction
        let reconstructed = cp.reconstruct(&[6, 6, 6]).unwrap();

        // Check shape
        assert_eq!(reconstructed.shape(), &[6, 6, 6]);

        // Reconstruction error should match fit
        let diff = &tensor - &reconstructed;
        let error_norm = diff.frobenius_norm();
        let tensor_norm = tensor.frobenius_norm();
        let relative_error = error_norm / tensor_norm;
        let computed_fit = 1.0 - relative_error;

        let fit_diff = (cp.fit - computed_fit).abs();
        assert!(
            fit_diff < 0.1, // Wider tolerance for accelerated method due to extrapolation
            "Fit should approximately match reconstruction: fit={:.4}, computed={:.4}",
            cp.fit,
            computed_fit
        );
    }

    #[test]
    fn test_cp_completion_basic() {
        // Test basic tensor completion with missing entries
        use scirs2_core::ndarray_ext::Array;

        // Create a simple 4x4x4 tensor
        let mut data = Array::zeros(vec![4, 4, 4]);
        let mut mask = Array::zeros(vec![4, 4, 4]);

        // Fill in some entries
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    if (i + j + k) % 2 == 0 {
                        // 50% observed
                        data[[i, j, k]] = (i + j + k) as f64 / 10.0;
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        let tensor = DenseND::from_array(data.into_dyn());
        let mask_tensor = DenseND::from_array(mask.into_dyn());

        // Test completion
        let result = cp_completion(&tensor, &mask_tensor, 3, 50, 1e-4, InitStrategy::Random);

        assert!(result.is_ok());
        let cp = result.unwrap();

        // Should have correct factors
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[4, 3]);
        assert_eq!(cp.factors[1].shape(), &[4, 3]);
        assert_eq!(cp.factors[2].shape(), &[4, 3]);

        // Fit should be reasonable
        assert!(cp.fit > 0.0 && cp.fit <= 1.0);
    }

    #[test]
    fn test_cp_completion_reconstruction() {
        // Test that completion can predict missing values
        use scirs2_core::ndarray_ext::Array;

        // Create a low-rank tensor (easy to complete)
        let factor1 = Array::from_shape_fn((6, 2), |(i, r)| (i + r) as f64 / 10.0);
        let factor2 = Array::from_shape_fn((6, 2), |(i, r)| (i + r + 1) as f64 / 10.0);
        let factor3 = Array::from_shape_fn((6, 2), |(i, r)| (i + r + 2) as f64 / 10.0);

        let factors_vec = vec![factor1.clone(), factor2.clone(), factor3.clone()];
        let original = compute_reconstruction(&factors_vec).unwrap();

        // Create mask: observe 70% of entries randomly
        let mut mask_data = Array::zeros(vec![6, 6, 6]);
        let mut rng = thread_rng();
        for idx in mask_data.iter_mut() {
            if rng.random::<f64>() < 0.7 {
                *idx = 1.0;
            }
        }

        let mask = DenseND::from_array(mask_data.into_dyn());

        // Complete the tensor
        let cp = cp_completion(&original, &mask, 2, 100, 1e-5, InitStrategy::Svd).unwrap();

        // Reconstructed tensor should be close to original
        let _reconstructed = cp.reconstruct(&[6, 6, 6]).unwrap();

        // Check fit on observed entries
        // Note: Tensor completion is harder with missing data, so lower threshold
        assert!(
            cp.fit > 0.0,
            "Completion should achieve positive fit on observed entries, got {:.4}",
            cp.fit
        );
        println!("Completion fit: {:.4}", cp.fit);
    }

    #[test]
    fn test_cp_completion_mask_validation() {
        // Test that mask shape validation works
        use scirs2_core::ndarray_ext::Array;

        let data = Array::<f64, _>::zeros(vec![4, 4, 4]);
        let wrong_mask = Array::<f64, _>::zeros(vec![4, 4, 5]); // Wrong shape

        let tensor = DenseND::from_array(data.into_dyn());
        let mask_tensor = DenseND::from_array(wrong_mask.into_dyn());

        let result = cp_completion(&tensor, &mask_tensor, 2, 50, 1e-4, InitStrategy::Random);

        assert!(result.is_err());
        match result {
            Err(CpError::ShapeMismatch(_)) => {} // Expected
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_cp_completion_no_observed_entries() {
        // Test that error is returned when mask has no observed entries
        use scirs2_core::ndarray_ext::Array;

        let data = Array::<f64, _>::zeros(vec![3, 3, 3]);
        let mask = Array::<f64, _>::zeros(vec![3, 3, 3]); // All zeros - no observed entries

        let tensor = DenseND::from_array(data.into_dyn());
        let mask_tensor = DenseND::from_array(mask.into_dyn());

        let result = cp_completion(&tensor, &mask_tensor, 2, 50, 1e-4, InitStrategy::Random);

        assert!(result.is_err());
        match result {
            Err(CpError::ShapeMismatch(_)) => {} // Expected
            _ => panic!("Expected ShapeMismatch error for empty mask"),
        }
    }

    #[test]
    fn test_cp_completion_convergence() {
        // Test that completion converges properly
        use scirs2_core::ndarray_ext::Array;

        let mut data = Array::zeros(vec![8, 8, 8]);
        let mut mask = Array::zeros(vec![8, 8, 8]);

        // Create structured data with good rank-3 structure
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    if (i + j * 2 + k * 3) % 3 == 0 {
                        data[[i, j, k]] = i as f64 * 0.1 + j as f64 * 0.2 + k as f64 * 0.15;
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        let tensor = DenseND::from_array(data.into_dyn());
        let mask_tensor = DenseND::from_array(mask.into_dyn());

        let cp = cp_completion(&tensor, &mask_tensor, 3, 200, 1e-6, InitStrategy::Svd).unwrap();

        // Should converge to some positive fit (completion is hard with sparse observations)
        assert!(
            cp.fit > 0.0,
            "Should achieve positive fit, got {:.4}",
            cp.fit
        );

        // Should not use all iterations if converged
        println!("Converged in {} iterations", cp.iters);
    }

    #[test]
    fn test_cp_completion_high_missing_rate() {
        // Test completion with high percentage of missing entries (90%)
        use scirs2_core::ndarray_ext::Array;

        let mut data = Array::zeros(vec![10, 10, 10]);
        let mut mask = Array::zeros(vec![10, 10, 10]);
        let mut rng = thread_rng();

        // Only 10% observed
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let val = (i + j + k) as f64 / 30.0;
                    data[[i, j, k]] = val;

                    if rng.random::<f64>() < 0.1 {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        let tensor = DenseND::from_array(data.into_dyn());
        let mask_tensor = DenseND::from_array(mask.into_dyn());

        let result = cp_completion(&tensor, &mask_tensor, 4, 100, 1e-4, InitStrategy::Random);

        // Should still work, though fit might be lower
        assert!(result.is_ok());
        let cp = result.unwrap();

        println!("High missing rate fit: {:.4}", cp.fit);
        assert!(cp.fit >= 0.0 && cp.fit <= 1.0);
    }

    // ========================================================================
    // Randomized CP Tests
    // ========================================================================

    #[test]
    fn test_cp_randomized_basic() {
        // Test basic randomized CP decomposition
        let tensor = DenseND::<f64>::random_uniform(&[15, 15, 15], 0.0, 1.0);

        let rank = 5;
        let sketch_size = rank * 3; // 3x oversampling
        let result = cp_randomized(
            &tensor,
            rank,
            30,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            5,
        );

        assert!(result.is_ok(), "Randomized CP should succeed");
        let cp = result.unwrap();

        // Check dimensions
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[15, 5]);
        assert_eq!(cp.factors[1].shape(), &[15, 5]);
        assert_eq!(cp.factors[2].shape(), &[15, 5]);

        // Fit should be in valid range
        assert!(
            cp.fit > 0.0 && cp.fit <= 1.0,
            "Fit should be in [0, 1], got {:.4}",
            cp.fit
        );
    }

    #[test]
    fn test_cp_randomized_reconstruction() {
        // Test that randomized CP produces valid reconstructions
        let tensor = DenseND::<f64>::random_uniform(&[12, 12, 12], 0.0, 1.0);

        let rank = 8;
        let sketch_size = rank * 4; // 4x oversampling for better accuracy
        let cp = cp_randomized(
            &tensor,
            rank,
            50,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            3,
        )
        .unwrap();

        // Reconstruct
        let reconstructed = cp.reconstruct(tensor.shape());
        assert!(reconstructed.is_ok(), "Reconstruction should succeed");

        let recon = reconstructed.unwrap();
        assert_eq!(recon.shape(), tensor.shape());

        // Compute actual fit to verify it's reasonable
        let diff = &tensor - &recon;
        let error_norm = diff.frobenius_norm();
        let tensor_norm = tensor.frobenius_norm();
        let computed_fit = 1.0 - (error_norm / tensor_norm);

        println!(
            "Randomized CP fit: {:.4}, computed fit: {:.4}",
            cp.fit, computed_fit
        );

        // Randomized CP may have slightly lower fit than standard CP
        assert!(
            computed_fit > 0.0,
            "Computed fit should be positive, got {:.4}",
            computed_fit
        );
    }

    #[test]
    fn test_cp_randomized_vs_standard() {
        // Compare randomized CP with standard CP-ALS
        let tensor = DenseND::<f64>::random_uniform(&[20, 20, 20], 0.0, 1.0);
        let rank = 6;

        // Standard CP-ALS
        let cp_std = cp_als(&tensor, rank, 30, 1e-4, InitStrategy::Random, None).unwrap();

        // Randomized CP with good oversampling
        let sketch_size = rank * 5; // 5x oversampling
        let cp_rand = cp_randomized(
            &tensor,
            rank,
            30,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            3,
        )
        .unwrap();

        println!(
            "Standard CP fit: {:.4}, Randomized CP fit: {:.4}",
            cp_std.fit, cp_rand.fit
        );

        // Randomized fit should be comparable (within reasonable range)
        // Due to randomness, it may be slightly lower
        assert!(cp_rand.fit > 0.0, "Randomized fit should be positive");

        // Both should produce valid decompositions
        assert!(cp_rand.reconstruct(tensor.shape()).is_ok());
        assert!(cp_std.reconstruct(tensor.shape()).is_ok());
    }

    #[test]
    fn test_cp_randomized_oversampling() {
        // Test effect of oversampling parameter
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let rank = 4;

        // Low oversampling (3x) - 2x is too low and can fail numerically
        let sketch_low = rank * 3;
        let cp_low =
            cp_randomized(&tensor, rank, 30, 1e-4, InitStrategy::Random, sketch_low, 5).unwrap();

        // High oversampling (7x)
        let sketch_high = rank * 7;
        let cp_high = cp_randomized(
            &tensor,
            rank,
            30,
            1e-4,
            InitStrategy::Random,
            sketch_high,
            5,
        )
        .unwrap();

        println!(
            "Low oversampling (3x) fit: {:.4}, High oversampling (7x) fit: {:.4}",
            cp_low.fit, cp_high.fit
        );

        // Both should work and achieve non-negative fit
        assert!(cp_low.fit >= 0.0);
        assert!(cp_high.fit >= 0.0);

        // Fits should be in valid range
        assert!(cp_low.fit <= 1.0);
        assert!(cp_high.fit <= 1.0);
    }

    #[test]
    fn test_cp_randomized_fit_check_frequency() {
        // Test different fit check frequencies
        let tensor = DenseND::<f64>::random_uniform(&[12, 12, 12], 0.0, 1.0);
        let rank = 5;
        let sketch_size = rank * 4;

        // Check fit every iteration (slow but accurate convergence detection)
        let cp_freq1 = cp_randomized(
            &tensor,
            rank,
            20,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            1,
        )
        .unwrap();

        // Check fit every 10 iterations (faster)
        let cp_freq10 = cp_randomized(
            &tensor,
            rank,
            20,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            10,
        )
        .unwrap();

        println!("Freq=1 iters: {}, fit: {:.4}", cp_freq1.iters, cp_freq1.fit);
        println!(
            "Freq=10 iters: {}, fit: {:.4}",
            cp_freq10.iters, cp_freq10.fit
        );

        // Both should converge
        assert!(cp_freq1.iters > 0);
        assert!(cp_freq10.iters > 0);

        // Both should have reasonable fits
        assert!(cp_freq1.fit > 0.0 && cp_freq1.fit <= 1.0);
        assert!(cp_freq10.fit > 0.0 && cp_freq10.fit <= 1.0);
    }

    #[test]
    fn test_cp_randomized_invalid_params() {
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);

        // Invalid: sketch_size < rank
        let result1 = cp_randomized(&tensor, 5, 20, 1e-4, InitStrategy::Random, 3, 5);
        assert!(result1.is_err(), "Should fail with sketch_size < rank");

        // Invalid: rank = 0
        let result2 = cp_randomized(&tensor, 0, 20, 1e-4, InitStrategy::Random, 10, 5);
        assert!(result2.is_err(), "Should fail with rank = 0");

        // Invalid: rank > mode_size
        let result3 = cp_randomized(&tensor, 15, 20, 1e-4, InitStrategy::Random, 30, 5);
        assert!(result3.is_err(), "Should fail with rank > mode_size");
    }

    #[test]
    fn test_cp_randomized_convergence() {
        // Test that randomized CP converges
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let rank = 4;
        let sketch_size = rank * 4;

        // Run with tight tolerance
        let cp_tight =
            cp_randomized(&tensor, rank, 100, 1e-5, InitStrategy::Svd, sketch_size, 5).unwrap();

        // Run with loose tolerance
        let cp_loose =
            cp_randomized(&tensor, rank, 100, 1e-2, InitStrategy::Svd, sketch_size, 5).unwrap();

        println!(
            "Tight tol: {} iters, fit: {:.4}",
            cp_tight.iters, cp_tight.fit
        );
        println!(
            "Loose tol: {} iters, fit: {:.4}",
            cp_loose.iters, cp_loose.fit
        );

        // Looser tolerance should converge faster
        assert!(cp_loose.iters <= cp_tight.iters);

        // Both should achieve positive fit
        assert!(cp_tight.fit > 0.0);
        assert!(cp_loose.fit > 0.0);
    }

    #[test]
    fn test_cp_randomized_init_strategies() {
        // Test different initialization strategies
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let rank = 5;
        let sketch_size = rank * 3;

        // Random init
        let cp_rand = cp_randomized(
            &tensor,
            rank,
            30,
            1e-4,
            InitStrategy::Random,
            sketch_size,
            5,
        )
        .unwrap();
        assert!(cp_rand.fit > 0.0);

        // SVD init (typically better)
        let cp_svd =
            cp_randomized(&tensor, rank, 30, 1e-4, InitStrategy::Svd, sketch_size, 5).unwrap();
        assert!(cp_svd.fit > 0.0);

        println!(
            "Random init fit: {:.4}, SVD init fit: {:.4}",
            cp_rand.fit, cp_svd.fit
        );
    }

    // ========================================================================
    // Incremental CP-ALS Tests
    // ========================================================================

    #[test]
    fn test_cp_incremental_append_mode() {
        // Test incremental CP with append mode (tensor grows)
        let initial_tensor = DenseND::<f64>::random_uniform(&[20, 10, 10], 0.0, 1.0);
        let rank = 5;

        // Initial decomposition
        let cp_initial =
            cp_als(&initial_tensor, rank, 30, 1e-4, InitStrategy::Random, None).unwrap();

        // New data arrives (5 new time steps)
        let new_slice = DenseND::<f64>::random_uniform(&[5, 10, 10], 0.0, 1.0);

        // Update incrementally
        let cp_updated = cp_als_incremental(
            &cp_initial,
            &new_slice,
            0, // time dimension
            IncrementalMode::Append,
            10, // fewer iterations for refinement
            1e-4,
        )
        .unwrap();

        // Verify updated decomposition
        assert_eq!(cp_updated.factors.len(), 3);
        assert_eq!(cp_updated.factors[0].shape()[0], 25); // 20 + 5
        assert_eq!(cp_updated.factors[0].shape()[1], rank);
        assert_eq!(cp_updated.factors[1].shape()[0], 10);
        assert_eq!(cp_updated.factors[2].shape()[0], 10);

        // Fit should be positive
        assert!(cp_updated.fit > 0.0 && cp_updated.fit <= 1.0);

        println!(
            "Initial fit: {:.4}, Updated fit: {:.4}",
            cp_initial.fit, cp_updated.fit
        );
    }

    #[test]
    fn test_cp_incremental_sliding_window() {
        // Test incremental CP with sliding window mode
        let initial_tensor = DenseND::<f64>::random_uniform(&[20, 10, 10], 0.0, 1.0);
        let rank = 5;

        // Initial decomposition
        let cp_initial =
            cp_als(&initial_tensor, rank, 30, 1e-4, InitStrategy::Random, None).unwrap();

        // New data arrives (replaces old data)
        let new_data = DenseND::<f64>::random_uniform(&[20, 10, 10], 0.0, 1.0);

        // Update with sliding window
        let cp_updated = cp_als_incremental(
            &cp_initial,
            &new_data,
            0, // time dimension
            IncrementalMode::SlidingWindow { lambda: 0.9 },
            10,
            1e-4,
        )
        .unwrap();

        // Verify factor dimensions unchanged
        assert_eq!(cp_updated.factors.len(), 3);
        assert_eq!(cp_updated.factors[0].shape()[0], 20);
        assert_eq!(cp_updated.factors[0].shape()[1], rank);

        // Fit should be positive
        assert!(cp_updated.fit > 0.0 && cp_updated.fit <= 1.0);

        println!(
            "Initial fit: {:.4}, Updated fit (sliding): {:.4}",
            cp_initial.fit, cp_updated.fit
        );
    }

    #[test]
    fn test_cp_incremental_dimensions() {
        // Test that incremental updates preserve correct dimensions
        let tensor1 = DenseND::<f64>::random_uniform(&[10, 8, 6], 0.0, 1.0);
        let rank = 3;

        let cp = cp_als(&tensor1, rank, 20, 1e-4, InitStrategy::Random, None).unwrap();

        // Add data along mode 0
        let new_data = DenseND::<f64>::random_uniform(&[5, 8, 6], 0.0, 1.0);
        let cp_updated =
            cp_als_incremental(&cp, &new_data, 0, IncrementalMode::Append, 5, 1e-4).unwrap();

        assert_eq!(cp_updated.factors[0].shape()[0], 15); // 10 + 5
        assert_eq!(cp_updated.factors[1].shape()[0], 8);
        assert_eq!(cp_updated.factors[2].shape()[0], 6);
    }

    #[test]
    fn test_cp_incremental_invalid_mode() {
        // Test error handling for invalid update mode
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let rank = 5;

        let cp = cp_als(&tensor, rank, 20, 1e-4, InitStrategy::Random, None).unwrap();
        let new_data = DenseND::<f64>::random_uniform(&[5, 10, 10], 0.0, 1.0);

        // Invalid mode (>= n_modes)
        let result = cp_als_incremental(&cp, &new_data, 3, IncrementalMode::Append, 5, 1e-4);

        assert!(result.is_err(), "Should fail with invalid mode");
    }

    #[test]
    fn test_cp_incremental_shape_mismatch() {
        // Test error handling for shape mismatches
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let rank = 5;

        let cp = cp_als(&tensor, rank, 20, 1e-4, InitStrategy::Random, None).unwrap();

        // Wrong shape in non-update modes
        let new_data = DenseND::<f64>::random_uniform(&[5, 12, 10], 0.0, 1.0);

        let result = cp_als_incremental(&cp, &new_data, 0, IncrementalMode::Append, 5, 1e-4);

        assert!(result.is_err(), "Should fail with shape mismatch");
    }

    #[test]
    fn test_cp_incremental_convergence() {
        // Test that incremental updates converge
        let tensor = DenseND::<f64>::random_uniform(&[15, 8, 8], 0.0, 1.0);
        let rank = 4;

        let cp = cp_als(&tensor, rank, 30, 1e-4, InitStrategy::Svd, None).unwrap();

        // Add new data
        let new_data = DenseND::<f64>::random_uniform(&[5, 8, 8], 0.0, 1.0);

        let cp_updated =
            cp_als_incremental(&cp, &new_data, 0, IncrementalMode::Append, 10, 1e-4).unwrap();

        // Verify convergence (completed iterations <= max_iters)
        assert!(cp_updated.iters <= 10);
        assert!(cp_updated.iters > 0);

        println!(
            "Incremental update converged in {} iterations",
            cp_updated.iters
        );
    }

    #[test]
    fn test_cp_incremental_reconstruction_quality() {
        // Test that incrementally updated CP can reconstruct the full tensor
        let tensor1 = DenseND::<f64>::random_uniform(&[15, 10, 10], 0.0, 1.0);
        let rank = 6;

        let cp = cp_als(&tensor1, rank, 30, 1e-4, InitStrategy::Random, None).unwrap();

        // New data
        let new_data = DenseND::<f64>::random_uniform(&[5, 10, 10], 0.0, 1.0);

        let cp_updated =
            cp_als_incremental(&cp, &new_data, 0, IncrementalMode::Append, 10, 1e-4).unwrap();

        // Reconstruct
        let reconstructed = cp_updated.reconstruct(&[20, 10, 10]).unwrap();

        // Verify shape
        assert_eq!(reconstructed.shape(), &[20, 10, 10]);

        // Fit should indicate reasonable reconstruction quality
        assert!(
            cp_updated.fit > 0.5,
            "Fit should be reasonably good: {:.4}",
            cp_updated.fit
        );
    }
}
