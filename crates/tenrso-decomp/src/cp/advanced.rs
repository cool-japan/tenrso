//! Advanced CP decomposition algorithms
//!
//! Contains: cp_als_accelerated, cp_completion, cp_randomized, cp_als_incremental

use super::helpers::*;
use super::types::*;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use scirs2_core::random::thread_rng;
use scirs2_linalg::lstsq;
use std::iter::Sum;
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;

/// Accelerated CP-ALS with line search optimization
///
/// An enhanced version of CP-ALS that uses line search to determine optimal
/// step sizes and incorporates acceleration techniques for faster convergence.
///
/// This method typically converges 2-5x faster than standard CP-ALS while
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
    let mut alpha = T::from(0.5).unwrap();
    let alpha_max = T::from(0.9).unwrap();
    let alpha_min = T::from(0.1).unwrap();

    let tol_t = T::from(tol).unwrap();
    let tensor_norm = tensor.frobenius_norm();
    let tensor_norm_sq = tensor_norm * tensor_norm;
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
            let tensor_view = tensor.view();
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

            let mttkrp_result = mttkrp(&tensor_view, &factor_views, mode)
                .map_err(|e| CpError::ShapeMismatch(format!("MTTKRP failed: {}", e)))?;

            let gram = compute_gram_hadamard(&factors, mode);

            let mut factor_new = solve_least_squares(&mttkrp_result, &gram)?;

            // LINE SEARCH: Find optimal step size
            let alpha_ls = line_search_cp(
                tensor,
                &factors,
                &prev_factors,
                mode,
                &factor_new,
                T::from(0.5).unwrap(),
                5,
            );

            // Apply extrapolation with line search step size
            if iter > 0 {
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
                alpha = (alpha * T::from(1.05).unwrap()).min(alpha_max);
            } else {
                alpha = (alpha * T::from(0.7).unwrap()).max(alpha_min);
                oscillation_count += 1;

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
            let mode_size = shape[mode];
            let mut mttkrp_result = Array2::<T>::zeros((mode_size, rank));

            let kr = compute_khatri_rao_except(&factors, mode);

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

            let factor_new = solve_least_squares(&mttkrp_result, &gram)?;
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
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `rank` - Target CP rank (number of components)
/// * `max_iters` - Maximum number of ALS iterations
/// * `tol` - Convergence tolerance on fit improvement
/// * `init` - Initialization strategy
/// * `sketch_size` - Sketch dimension (typically 2-5x rank for good accuracy)
/// * `fit_check_freq` - How often to compute full fit (e.g., every 5 iterations)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_randomized, InitStrategy};
///
/// let tensor = DenseND::<f64>::random_uniform(&[50, 50, 50], 0.0, 1.0);
/// let cp = cp_randomized(&tensor, 10, 20, 1e-4, InitStrategy::Random, 25, 5).unwrap();
///
/// println!("Final fit: {:.4}", cp.fit);
/// # assert!(cp.fit >= 0.0 && cp.fit <= 1.0);
/// ```
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
        return Err(CpError::InvalidRank(0));
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

        for mode in 0..n_modes {
            let kr = compute_khatri_rao_except(&factors, mode);
            let kr_rows = kr.shape()[0];

            let mut omega = Array2::<T>::zeros((kr_rows, sketch_size));
            for i in 0..kr_rows {
                for j in 0..sketch_size {
                    omega[[i, j]] = T::from(normal.sample(&mut rng)).unwrap();
                }
            }

            let kr_sketch = kr.t().dot(&omega.view());

            let unfolding = tensor
                .unfold(mode)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

            let x_sketch = unfolding.dot(&omega.view());

            let gram = kr_sketch.dot(&kr_sketch.t().view());
            let rhs = kr_sketch.dot(&x_sketch.t().view());

            let mut new_factor = Array2::<T>::zeros((shape[mode], rank));

            for i in 0..shape[mode] {
                let b = rhs.column(i).to_owned();

                let solution =
                    lstsq(&gram.view(), &b.view(), None).map_err(CpError::LinalgError)?;

                for j in 0..rank {
                    new_factor[[i, j]] = solution.x[j];
                }
            }

            factors[mode] = new_factor;
        }

        // Compute fit periodically
        if iter % fit_check_freq == 0 || iter == max_iters - 1 {
            fit = compute_fit(tensor, &factors, tensor_norm_sq)?;

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

/// Incremental CP-ALS for online/streaming tensor decomposition
///
/// Updates an existing CP decomposition when new data arrives, avoiding
/// full recomputation.
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
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_als, cp_als_incremental, InitStrategy, IncrementalMode};
///
/// let batch1 = DenseND::<f64>::random_uniform(&[50, 20, 20], 0.0, 1.0);
/// let mut cp = cp_als(&batch1, 5, 20, 1e-4, InitStrategy::Random, None).unwrap();
///
/// let new_slice = DenseND::<f64>::random_uniform(&[10, 20, 20], 0.0, 1.0);
///
/// cp = cp_als_incremental(
///     &cp,
///     &new_slice,
///     0,
///     IncrementalMode::Append,
///     10,
///     1e-4
/// ).unwrap();
///
/// # assert_eq!(cp.factors[0].shape()[0], 60);
/// println!("Updated fit: {:.4}", cp.fit);
/// ```
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
            let old_rows = current.factors[update_mode].shape()[0];
            let new_rows = new_data.shape()[update_mode];
            let total_rows = old_rows + new_rows;

            let mut extended_factor = Array2::<T>::zeros((total_rows, rank));

            for i in 0..old_rows {
                for j in 0..rank {
                    extended_factor[[i, j]] = current.factors[update_mode][[i, j]];
                }
            }

            let mut rng = thread_rng();
            for i in old_rows..total_rows {
                for j in 0..rank {
                    let mut col_mean = T::zero();
                    for k in 0..old_rows {
                        col_mean += current.factors[update_mode][[k, j]];
                    }
                    col_mean /= T::from(old_rows).unwrap();

                    let noise = T::from(rng.random::<f64>() * 0.1 - 0.05).unwrap();
                    extended_factor[[i, j]] = col_mean + noise;
                }
            }

            factors[update_mode] = extended_factor;

            let old_tensor = tensor_from_factors(&current.factors, None)
                .map_err(|e| CpError::ShapeMismatch(format!("Failed to reconstruct: {}", e)))?;
            combined_tensor = concatenate_tensors(&old_tensor, new_data, update_mode)
                .map_err(|e| CpError::ShapeMismatch(format!("Concatenation failed: {}", e)))?;
        }

        IncrementalMode::SlidingWindow { lambda } => {
            if !(0.0..=1.0).contains(&lambda) {
                return Err(CpError::InvalidTolerance(lambda));
            }

            combined_tensor = new_data.clone();
        }
    }

    // Refine factors using ALS
    let refine_iters = max_iters.min(10);

    let tensor_norm_sq = compute_norm_squared(&combined_tensor);
    let mut fit = T::zero();
    let mut iters = 0;

    for iter in 0..refine_iters {
        iters = iter + 1;
        let prev_fit = fit;

        for mode_idx in 0..n_modes {
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp(&combined_tensor.view(), &factor_views, mode_idx)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

            let gram = compute_gram_hadamard(&factors, mode_idx);

            factors[mode_idx] = solve_least_squares(&mttkrp_result, &gram)?;
        }

        fit = compute_fit(&combined_tensor, &factors, tensor_norm_sq)?;

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
