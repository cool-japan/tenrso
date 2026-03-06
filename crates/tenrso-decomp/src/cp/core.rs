//! Core CP-ALS algorithms: standard and constrained
//!
//! Contains the main `cp_als` and `cp_als_constrained` functions.

use super::helpers::*;
use super::types::*;
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use std::iter::Sum;
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;

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
///
/// # Complexity
///
/// Time: O(I * R^2 * prod_i I_i) per iteration where I is max_iters
/// Space: O(N * Imax * R) for factor matrices
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_als, InitStrategy};
///
/// // Create a 10x10x10 tensor
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
/// - L1 regularization for sparsity-promoting decompositions
/// - Elastic net (combined L1 + L2) regularization
/// - Tikhonov regularization for smooth factor matrices
/// - Orthogonality constraints on factor matrices
///
/// # Non-negativity validation
///
/// When `constraints.nonnegative` is true and `constraints.validate_nonneg_input` is true,
/// the input tensor is validated to ensure all values are non-negative. This can be disabled
/// by setting `validate_nonneg_input = false`.
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

    // Validate regularization parameters
    constraints.regularization.validate()?;

    // Legacy l2_reg validation (when not using advanced regularization)
    if matches!(constraints.regularization, RegularizationType::None) && constraints.l2_reg < 0.0 {
        return Err(CpError::InvalidTolerance(constraints.l2_reg));
    }

    // Validate non-negative input tensor when constraint is active
    if constraints.nonnegative && constraints.validate_nonneg_input {
        validate_nonnegative(tensor)?;
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

    // Get effective regularization parameters
    let l2_lambda = constraints.effective_l2();
    let l1_lambda = constraints.effective_l1();

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

            // Step 3: Apply regularization to the Gram matrix
            match constraints.regularization {
                RegularizationType::Tikhonov { lambda, order } if order > 0 => {
                    // Apply Tikhonov with finite difference operator
                    let lambda_t = NumCast::from(lambda).unwrap();
                    apply_tikhonov_to_gram(&mut gram, lambda_t, order);
                }
                _ => {
                    // Apply L2 regularization (standard ridge on diagonal)
                    if l2_lambda > 0.0 {
                        let reg: T = NumCast::from(l2_lambda).unwrap();
                        for i in 0..gram.nrows() {
                            gram[[i, i]] += reg;
                        }
                    }
                }
            }

            // Step 4: Solve least squares
            factors[mode] = solve_least_squares(&mttkrp_result, &gram)?;

            // Step 5: Apply L1 regularization (soft-thresholding)
            if l1_lambda > 0.0 {
                let threshold: T = NumCast::from(l1_lambda).unwrap();
                soft_threshold(&mut factors[mode], threshold);
            }

            // Step 6: Apply constraints
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
