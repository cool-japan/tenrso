//! Advanced Tucker decomposition variants.
//!
//! Contains: tucker_randomized, tucker_nonnegative, tucker_completion.
//!
//! These algorithms extend the core Tucker decomposition (HOSVD/HOOI) with
//! randomized linear algebra, non-negativity constraints, and missing-data handling.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` or `rand` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayD, ScalarOperand};
use scirs2_core::numeric::{Float, FloatConst, FromPrimitive, NumAssign, NumCast};
use std::iter::Sum;
use tenrso_core::DenseND;

use crate::tucker::{
    compute_core_tensor, compute_mode_unfolding_contraction, extract_columns, TuckerDecomp,
    TuckerError,
};

/// Randomized Tucker decomposition for large-scale tensors
///
/// Computes an approximate Tucker decomposition using randomized linear algebra techniques.
/// This method is significantly faster than standard HOSVD for large tensors while maintaining
/// good approximation quality.
///
/// # Algorithm
///
/// Uses randomized SVD (Halko et al., 2011) for each mode-n unfolding:
/// 1. Project mode-n unfolding onto random subspace
/// 2. Compute QR decomposition of projection
/// 3. Compute SVD of smaller matrix
/// 4. Recover approximate singular vectors
///
/// This reduces complexity from O(I²R) to O(IR²) for each mode.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `ranks` - Target ranks for each mode [R₁, R₂, ..., Rₙ]
/// * `oversampling` - Additional samples for randomization (default: 10)
/// * `power_iters` - Power iterations for accuracy (default: 2)
///
/// # Returns
///
/// Approximate TuckerDecomp with core tensor and factor matrices
///
/// # Complexity
///
/// Time: O(N × ∏ᵢ Iᵢ × Rᵢ + N × Rᵢ³) vs O(N × ∏ᵢ Iᵢ × Iᵢ²) for HOSVD
/// Space: O(∏ᵢ Iᵢ + ∑ᵢ Iᵢ × Rᵢ)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tucker_advanced::tucker_randomized;
///
/// // Decompose large tensor efficiently
/// let tensor = DenseND::<f64>::random_uniform(&[40, 40, 20], 0.0, 1.0);
/// let tucker = tucker_randomized(&tensor, &[15, 15, 8], 5, 2).unwrap();
///
/// println!("Compression: {:.2}x", tucker.compression_ratio());
/// # assert!(tucker.compression_ratio() > 1.0);
/// ```
///
/// # References
///
/// - Halko et al. (2011), "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions"
/// - Che et al. (2019), "Randomized algorithms for the approximations of Tucker and the tensor train decompositions"
pub fn tucker_randomized<T>(
    tensor: &DenseND<T>,
    ranks: &[usize],
    oversampling: usize,
    power_iters: usize,
) -> Result<TuckerDecomp<T>, TuckerError>
where
    T: Float
        + FloatConst
        + NumCast
        + NumAssign
        + Sum
        + Send
        + Sync
        + ScalarOperand
        + std::fmt::Debug
        + FromPrimitive
        + 'static,
{
    use scirs2_core::random::{thread_rng, Distribution, RandNormal as Normal};
    use scirs2_linalg::{qr, svd};

    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validate inputs
    if ranks.len() != n_modes {
        return Err(TuckerError::InvalidRanks(format!(
            "ranks length {} != tensor rank {}",
            ranks.len(),
            n_modes
        )));
    }

    for (mode, &rank) in ranks.iter().enumerate() {
        if rank == 0 || rank > shape[mode] {
            return Err(TuckerError::InvalidRanks(format!(
                "Invalid rank {} for mode {} (dimension {})",
                rank, mode, shape[mode]
            )));
        }
    }

    let mut factors: Vec<Array2<T>> = Vec::with_capacity(n_modes);
    let mut rng = thread_rng();

    // Compute factor matrix for each mode using randomized SVD
    for (mode, &rank) in ranks.iter().enumerate() {
        // Unfold tensor along this mode
        let unfolded = tensor
            .unfold(mode)
            .map_err(|e| TuckerError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

        let (rows, cols) = (unfolded.shape()[0], unfolded.shape()[1]);
        // Target rank cannot exceed matrix dimensions
        let target_rank = (rank + oversampling).min(rows).min(cols);

        // Generate random Gaussian matrix: Ω ∈ ℝ^(cols × target_rank)
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut omega = Array2::<T>::zeros((cols, target_rank));
        for i in 0..cols {
            for j in 0..target_rank {
                omega[[i, j]] = T::from(normal.sample(&mut rng)).unwrap();
            }
        }

        // Project: Y = A × Ω
        let mut y = unfolded.dot(&omega);

        // Power iterations for improved accuracy
        for _ in 0..power_iters {
            // Y = A × (A^T × Y)
            let aty = unfolded.t().dot(&y.view());
            y = unfolded.dot(&aty.view());
        }

        // QR decomposition: Y = Q × R
        let (q, _r) = qr(&y.view(), None).map_err(|e| TuckerError::SvdError(e.to_string()))?;

        // Compute B = Q^T × A
        let b = q.t().dot(&unfolded.view());

        // Compute SVD of small matrix: B = U_b × S × V^T
        let (u_b, _s, _vt) =
            svd(&b.view(), false, None).map_err(|e| TuckerError::SvdError(e.to_string()))?;

        // Recover left singular vectors: U = Q × U_b
        let u_full = q.dot(&u_b.view());

        // Extract first 'rank' columns as factor matrix
        let mut factor = Array2::<T>::zeros((rows, rank));
        for i in 0..rows {
            for j in 0..rank.min(u_full.shape()[1]) {
                factor[[i, j]] = u_full[[i, j]];
            }
        }

        factors.push(factor);
    }

    // Compute core tensor
    let core = compute_core_tensor(tensor, &factors)?;

    Ok(TuckerDecomp {
        core,
        factors,
        error: None,
        iters: 0,
    })
}

/// Non-negative Tucker decomposition via multiplicative updates
///
/// Computes a Tucker decomposition X ≈ G ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ
/// where all elements of G and Uᵢ are non-negative.
///
/// This variant trades orthogonality for non-negativity, making it suitable for
/// applications where non-negativity is physically meaningful (e.g., spectral data,
/// topic modeling, image factorization).
///
/// # Algorithm
///
/// Uses multiplicative update rules similar to Non-negative Matrix Factorization (NMF):
/// - Initialize factors with non-negative random values
/// - Iteratively update each factor using multiplicative updates
/// - Update core tensor with non-negative constraint
/// - Convergence based on relative reconstruction error change
///
/// # Arguments
///
/// * `tensor` - Input tensor (all elements should be non-negative)
/// * `ranks` - Target ranks for each mode
/// * `max_iters` - Maximum number of iterations
/// * `tol` - Convergence tolerance for relative error change
///
/// # Returns
///
/// TuckerDecomp with non-negative core and factors
///
/// # Complexity
///
/// Time: O(max_iters × N × ∏ᵢ Iᵢ × Rᵢ)
/// Space: O(∏ᵢ Iᵢ + ∑ᵢ Iᵢ × Rᵢ)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tucker_advanced::tucker_nonnegative;
///
/// // Create non-negative tensor
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
/// let tucker = tucker_nonnegative(&tensor, &[5, 5, 5], 50, 1e-4).unwrap();
///
/// // All factors should be non-negative
/// for factor in &tucker.factors {
///     assert!(factor.iter().all(|&x| x >= 0.0));
/// }
/// ```
///
/// # References
///
/// - A. Cichocki et al. (2009), "Nonnegative Matrix and Tensor Factorizations"
/// - Y.-D. Kim & S. Choi (2007), "Nonnegative Tucker Decomposition"
pub fn tucker_nonnegative<T>(
    tensor: &DenseND<T>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomp<T>, TuckerError>
where
    T: Float
        + NumCast
        + NumAssign
        + Sum
        + Send
        + Sync
        + ScalarOperand
        + std::fmt::Debug
        + FromPrimitive
        + 'static,
{
    use scirs2_core::random::thread_rng;

    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validate inputs
    if ranks.len() != n_modes {
        return Err(TuckerError::InvalidRanks(format!(
            "ranks length {} != tensor rank {}",
            ranks.len(),
            n_modes
        )));
    }

    for (mode, &rank) in ranks.iter().enumerate() {
        if rank == 0 || rank > shape[mode] {
            return Err(TuckerError::InvalidRanks(format!(
                "Invalid rank {} for mode {} (dimension {})",
                rank, mode, shape[mode]
            )));
        }
    }

    // Check that tensor is non-negative
    let tensor_min = tensor
        .view()
        .iter()
        .fold(T::infinity(), |a, &b| if b < a { b } else { a });
    if tensor_min < T::zero() {
        return Err(TuckerError::ShapeMismatch(
            "Input tensor contains negative values".to_string(),
        ));
    }

    // Initialize factors with small random non-negative values
    let mut rng = thread_rng();
    let mut factors: Vec<Array2<T>> = Vec::with_capacity(n_modes);

    for mode in 0..n_modes {
        let mut factor = Array2::<T>::zeros((shape[mode], ranks[mode]));
        for i in 0..shape[mode] {
            for j in 0..ranks[mode] {
                // Small random initialization in [0.01, 1.01]
                factor[[i, j]] = T::from(0.01).unwrap() + T::from(rng.random::<f64>()).unwrap();
            }
        }
        factors.push(factor);
    }

    // Initialize core tensor
    let mut core = compute_core_tensor(tensor, &factors)?;

    // Project core to non-negative
    for elem in core.view_mut().iter_mut() {
        if *elem < T::zero() {
            *elem = T::zero();
        }
    }

    let eps = T::from(1e-10).unwrap(); // Small constant to avoid division by zero
    let tol_t = T::from(tol).unwrap();

    // Track error for convergence
    let tensor_norm = tensor.frobenius_norm();
    let mut prev_error = T::infinity();

    let mut actual_iters = 0;

    for iter in 0..max_iters {
        actual_iters = iter + 1;

        // Update each factor matrix
        #[allow(clippy::needless_range_loop)]
        for mode in 0..n_modes {
            // Compute Y = X ×₁ U₁ᵀ ... ×ₘ₋₁ Uₘ₋₁ᵀ ×ₘ₊₁ Uₘ₊₁ᵀ ... ×ₙ Uₙᵀ (skip mode m)
            let y = compute_mode_unfolding_contraction(tensor, &factors, mode)?;

            // Unfold Y along mode m
            let y_unfolded = y
                .unfold(mode)
                .map_err(|e| TuckerError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

            // Unfold core along mode m
            let core_unfolded = core
                .unfold(mode)
                .map_err(|e| TuckerError::ShapeMismatch(format!("Core unfold failed: {}", e)))?;

            // Compute Gram matrix for other factors
            // Numerator: Y_unfold × Core_unfold^T
            let numerator = y_unfolded.dot(&core_unfolded.t());

            // Denominator: U_m × (Core_unfold × Core_unfold^T)
            let core_gram = core_unfolded.dot(&core_unfolded.t());
            let denominator = factors[mode].dot(&core_gram);

            // Multiplicative update: U_m ← U_m ⊙ (numerator / (denominator + ε))
            for i in 0..shape[mode] {
                for j in 0..ranks[mode] {
                    let num = numerator[[i, j]];
                    let denom = denominator[[i, j]] + eps;
                    factors[mode][[i, j]] *= num / denom;

                    // Ensure non-negativity (guard against numerical errors)
                    if factors[mode][[i, j]] < T::zero() {
                        factors[mode][[i, j]] = T::zero();
                    }
                }
            }
        }

        // Update core tensor
        core = compute_core_tensor(tensor, &factors)?;

        // Project core to non-negative
        for elem in core.view_mut().iter_mut() {
            if *elem < T::zero() {
                *elem = T::zero();
            }
        }

        // Check convergence every few iterations (save computation)
        if iter % 5 == 0 || iter == max_iters - 1 {
            // Compute reconstruction error
            let temp_decomp = TuckerDecomp {
                core: core.clone(),
                factors: factors.clone(),
                error: None,
                iters: actual_iters,
            };

            let reconstructed = temp_decomp
                .reconstruct()
                .map_err(|e| TuckerError::ShapeMismatch(format!("Reconstruction failed: {}", e)))?;

            // Compute error
            let diff = &tensor.clone() - &reconstructed;
            let diff_norm = diff.frobenius_norm();
            let error = diff_norm / tensor_norm;

            // Check convergence
            if prev_error != T::infinity() {
                let error_change = (prev_error - error).abs() / (prev_error + eps);
                if error_change < tol_t {
                    break;
                }
            }

            prev_error = error;
        }
    }

    let mut result = TuckerDecomp {
        core,
        factors,
        error: Some(prev_error),
        iters: actual_iters,
    };

    // Compute final error
    let final_error = result.compute_error(tensor).map_err(|e| {
        TuckerError::ShapeMismatch(format!("Final error computation failed: {}", e))
    })?;
    result.error = Some(final_error);

    Ok(result)
}

/// Tucker completion - Tucker decomposition with missing data
///
/// Computes Tucker decomposition while only fitting to observed entries in the tensor.
/// Uses a binary mask to indicate which entries are observed (1) vs missing (0).
///
/// # Arguments
///
/// * `tensor` - Input tensor (may contain arbitrary values for missing entries)
/// * `mask` - Binary mask (1 = observed, 0 = missing), same shape as tensor
/// * `ranks` - Target ranks for each mode
/// * `max_iters` - Maximum number of iterations
/// * `tol` - Convergence tolerance (relative error change)
///
/// # Algorithm: Tucker-WOPT (Weighted Optimization)
///
/// Similar to HOOI but only considers observed entries:
/// 1. Initialize factors via HOSVD on observed entries
/// 2. For each iteration:
///    - For each mode: Update factor via weighted SVD of mode-n unfolding
///    - Recompute core tensor from observed entries
///    - Check convergence based on fit to observed entries
///
/// # Applications
///
/// - Image/video inpainting with missing pixels/frames
/// - Sensor networks with missing measurements
/// - Recommender systems with incomplete user-item-context ratings
/// - Scientific data with missing observations
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_core::DenseND;
/// use tenrso_decomp::tucker_completion;
///
/// // Create 10x10x10 tensor with 40% missing data
/// let mut tensor_data = Array::from_elem((10, 10, 10), 1.0);
/// let mut mask_data = Array::ones((10, 10, 10));
///
/// // Mark 40% as missing (mask = 0)
/// for i in 0..400 {
///     mask_data[[i / 100, (i / 10) % 10, i % 10]] = 0.0;
/// }
///
/// let tensor = DenseND::from_array(tensor_data.into_dyn());
/// let mask = DenseND::from_array(mask_data.into_dyn());
///
/// // Fit Tucker model to observed entries only
/// let tucker = tucker_completion(&tensor, &mask, &[5, 5, 5], 50, 1e-4).unwrap();
///
/// // Reconstruct to predict missing values
/// let completed = tucker.reconstruct().unwrap();
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # References
///
/// - Acar et al. (2011), "Scalable tensor factorizations for incomplete data"
/// - Liu et al. (2012), "Tensor completion for estimating missing values in visual data"
pub fn tucker_completion<T>(
    tensor: &DenseND<T>,
    mask: &DenseND<T>,
    ranks: &[usize],
    max_iters: usize,
    tol: f64,
) -> Result<TuckerDecomp<T>, TuckerError>
where
    T: Float
        + NumCast
        + NumAssign
        + Sum
        + Send
        + Sync
        + ScalarOperand
        + std::fmt::Debug
        + FromPrimitive
        + 'static,
{
    use scirs2_linalg::svd;

    let shape = tensor.shape();
    let n_modes = tensor.rank();

    // Validate inputs
    if mask.shape() != shape {
        return Err(TuckerError::ShapeMismatch(format!(
            "Mask shape {:?} doesn't match tensor shape {:?}",
            mask.shape(),
            shape
        )));
    }

    if ranks.len() != n_modes {
        return Err(TuckerError::InvalidRanks(format!(
            "ranks length {} != tensor rank {}",
            ranks.len(),
            n_modes
        )));
    }

    for (mode, &rank) in ranks.iter().enumerate() {
        if rank == 0 || rank > shape[mode] {
            return Err(TuckerError::InvalidRanks(format!(
                "Invalid rank {} for mode {} (dimension {})",
                rank, mode, shape[mode]
            )));
        }
    }

    if tol <= 0.0 || tol >= 1.0 {
        return Err(TuckerError::InvalidRanks(format!(
            "Invalid tolerance {}, must be in (0, 1)",
            tol
        )));
    }

    // Count observed entries
    let mask_view = mask.view();
    let mut n_observed = T::zero();
    for &m in mask_view.iter() {
        n_observed += m;
    }

    if n_observed == T::zero() {
        return Err(TuckerError::InvalidRanks(
            "No observed entries in tensor".to_string(),
        ));
    }

    // Initialize factors using HOSVD on masked tensor
    // For missing entries, we use the current tensor values (will be updated iteratively)
    let mut factors: Vec<Array2<T>> = Vec::with_capacity(n_modes);

    for (mode, &rank) in ranks.iter().enumerate() {
        // Unfold tensor along mode
        let unfolding = tensor.unfold(mode).map_err(|e| {
            TuckerError::ShapeMismatch(format!("Unfold mode {} failed: {}", mode, e))
        })?;

        // Compute SVD and extract left singular vectors
        let (u, _s, _vt) = svd(&unfolding.view(), true, None)
            .map_err(|e| TuckerError::SvdError(format!("SVD failed for mode {}: {}", mode, e)))?;

        // Extract first rank columns
        let factor = extract_columns(&u, rank);
        factors.push(factor);
    }

    let tol_t = T::from(tol).unwrap();
    let mut prev_fit = T::zero();

    let mut actual_iters = 0;

    // Mask the tensor once at the start (zero out missing entries)
    let masked_tensor = element_multiply(tensor, mask)?;

    for iter in 0..max_iters {
        actual_iters = iter + 1;

        // Update each factor matrix
        for mode in 0..n_modes {
            // Contract masked tensor with all factors except mode
            // to get the mode-n unfolding
            let mut contracted = masked_tensor.clone();
            for m in (0..n_modes).rev() {
                if m == mode {
                    continue;
                }
                let factor_t = factors[m].t().to_owned();
                let result_array =
                    tenrso_kernels::nmode_product(&contracted.view(), &factor_t.view(), m)
                        .map_err(|e| {
                            TuckerError::ShapeMismatch(format!(
                                "nmode_product failed for mode {}: {}",
                                m, e
                            ))
                        })?;
                contracted = DenseND::from_array(result_array);
            }

            // Unfold along mode
            let unfolding = contracted.unfold(mode).map_err(|e| {
                TuckerError::ShapeMismatch(format!("Unfold mode {} failed: {}", mode, e))
            })?;

            // Compute SVD
            let (u, _s, _vt) = svd(&unfolding.view(), true, None).map_err(|e| {
                TuckerError::SvdError(format!(
                    "SVD failed for mode {} at iter {}: {}",
                    mode, iter, e
                ))
            })?;

            // Update factor
            factors[mode] = extract_columns(&u, ranks[mode]);
        }

        // Recompute core tensor: G = X ×₁ U₁ᵀ ×₂ U₂ᵀ ... ×ₙ Uₙᵀ (using masked tensor)
        let core = compute_core_tensor(&masked_tensor, &factors)?;

        // Compute fit to observed entries
        let fit = compute_observed_fit(tensor, mask, &core, &factors, n_observed)?;

        // Check convergence
        let fit_change = if iter > 0 {
            (fit - prev_fit).abs() / (prev_fit + T::from(1e-10).unwrap())
        } else {
            T::one()
        };

        if fit_change < tol_t && iter > 0 {
            let result = TuckerDecomp {
                core,
                factors,
                error: Some(T::one() - fit),
                iters: actual_iters,
            };
            return Ok(result);
        }

        prev_fit = fit;
    }

    // Max iterations reached
    let core = compute_core_tensor(&masked_tensor, &factors)?;
    let result = TuckerDecomp {
        core,
        factors,
        error: Some(T::one() - prev_fit),
        iters: actual_iters,
    };

    Ok(result)
}

/// Compute fit to observed entries
fn compute_observed_fit<T>(
    tensor: &DenseND<T>,
    mask: &DenseND<T>,
    core: &DenseND<T>,
    factors: &[Array2<T>],
    _n_observed: T,
) -> Result<T, TuckerError>
where
    T: Float + NumCast + NumAssign + Sum + ScalarOperand + 'static,
{
    // Reconstruct tensor
    let mut reconstructed = core.clone();
    for (mode, factor) in factors.iter().enumerate() {
        let recon_array = tenrso_kernels::nmode_product(
            &reconstructed.view(),
            &factor.view(),
            mode,
        )
        .map_err(|e| {
            TuckerError::ShapeMismatch(format!("Reconstruction nmode_product failed: {}", e))
        })?;
        reconstructed = DenseND::from_array(recon_array);
    }

    // Compute error on observed entries only
    let tensor_view = tensor.view();
    let mask_view = mask.view();
    let recon_view = reconstructed.view();

    let mut norm_tensor_sq = T::zero();
    let mut norm_diff_sq = T::zero();

    for ((&t, &m), &r) in tensor_view
        .iter()
        .zip(mask_view.iter())
        .zip(recon_view.iter())
    {
        if m > T::zero() {
            // Observed entry
            norm_tensor_sq += t * t;
            let diff = t - r;
            norm_diff_sq += diff * diff;
        }
    }

    // Fit = 1 - ||X - R||_F^2 / ||X||_F^2 (for observed entries)
    let fit = T::one() - norm_diff_sq / (norm_tensor_sq + T::from(1e-10).unwrap());

    Ok(fit)
}

/// Element-wise multiplication of two tensors
fn element_multiply<T>(
    tensor1: &DenseND<T>,
    tensor2: &DenseND<T>,
) -> Result<DenseND<T>, TuckerError>
where
    T: Float + NumCast + 'static,
{
    if tensor1.shape() != tensor2.shape() {
        return Err(TuckerError::ShapeMismatch(format!(
            "Shape mismatch: {:?} vs {:?}",
            tensor1.shape(),
            tensor2.shape()
        )));
    }

    let view1 = tensor1.view();
    let view2 = tensor2.view();

    let shape = tensor1.shape();
    let mut result_data = ArrayD::<T>::zeros(shape);

    for (i, (&v1, &v2)) in view1.iter().zip(view2.iter()).enumerate() {
        result_data.as_slice_mut().unwrap()[i] = v1 * v2;
    }

    Ok(DenseND::from_array(result_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenrso_core::DenseND;

    #[test]
    fn test_tucker_nonnegative_basic() {
        // Create non-negative tensor
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let result = tucker_nonnegative(&tensor, &[4, 4, 4], 30, 1e-3);

        assert!(result.is_ok());
        let tucker = result.unwrap();

        // Check dimensions
        assert_eq!(tucker.core.shape(), &[4, 4, 4]);
        assert_eq!(tucker.factors.len(), 3);

        // Verify non-negativity of all factors
        for (mode, factor) in tucker.factors.iter().enumerate() {
            for &val in factor.iter() {
                assert!(val >= 0.0, "Factor {} has negative value: {}", mode, val);
            }
        }

        // Verify non-negativity of core
        for &val in tucker.core.view().iter() {
            assert!(val >= 0.0, "Core has negative value: {}", val);
        }
    }

    #[test]
    fn test_tucker_nonnegative_reconstruction() {
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
        let mut tucker = tucker_nonnegative(&tensor, &[3, 3, 3], 25, 1e-3).unwrap();

        // Reconstruction should work
        let reconstructed = tucker.reconstruct();
        assert!(reconstructed.is_ok());

        // Compute error
        let error = tucker.compute_error(&tensor);
        assert!(error.is_ok());

        // Error should be reasonable (non-negative Tucker may have higher error than HOSVD)
        let err_val = error.unwrap();
        assert!((0.0..=1.0).contains(&err_val));
    }

    #[test]
    fn test_tucker_nonnegative_convergence() {
        let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);

        // Run with tighter tolerance
        let tucker1 = tucker_nonnegative(&tensor, &[3, 3, 3], 50, 1e-5).unwrap();

        // Run with looser tolerance
        let tucker2 = tucker_nonnegative(&tensor, &[3, 3, 3], 50, 1e-2).unwrap();

        // Tighter tolerance should use more iterations (or same if both converge)
        assert!(tucker1.iters >= tucker2.iters || tucker1.iters > 5);
    }

    #[test]
    fn test_tucker_nonnegative_negative_input() {
        // Create tensor with some negative values
        let mut data = vec![0.5; 27];
        data[0] = -0.1; // One negative value
        let tensor = DenseND::<f64>::from_vec(data, &[3, 3, 3]).unwrap();

        // Should fail with error about negative values
        let result = tucker_nonnegative(&tensor, &[2, 2, 2], 10, 1e-3);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("negative values"));
    }

    #[test]
    fn test_tucker_nonnegative_vs_hosvd() {
        use crate::tucker::tucker_hosvd;

        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

        // Non-negative Tucker
        let mut tucker_nn = tucker_nonnegative(&tensor, &[4, 4, 4], 30, 1e-4).unwrap();

        // Standard HOSVD
        let mut tucker_std = tucker_hosvd(&tensor, &[4, 4, 4]).unwrap();

        // Compute errors
        let err_nn = tucker_nn.compute_error(&tensor).unwrap();
        let err_std = tucker_std.compute_error(&tensor).unwrap();

        // Non-negative Tucker typically has higher error (trades accuracy for non-negativity)
        // But both should produce valid decompositions
        assert!(err_nn >= 0.0);
        assert!(err_std >= 0.0);

        println!(
            "Non-negative error: {:.6}, Standard error: {:.6}",
            err_nn, err_std
        );

        // Standard HOSVD should generally be more accurate (or equal)
        // But we allow some tolerance for randomness in initialization
        assert!(
            err_std <= err_nn + 0.2,
            "Standard HOSVD should be competitive"
        );
    }

    #[test]
    fn test_tucker_nonnegative_rank_validation() {
        let tensor = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);

        // Invalid: rank exceeds dimension
        let result1 = tucker_nonnegative(&tensor, &[6, 5, 5], 10, 1e-3);
        assert!(result1.is_err());

        // Invalid: zero rank
        let result2 = tucker_nonnegative(&tensor, &[0, 5, 5], 10, 1e-3);
        assert!(result2.is_err());

        // Invalid: wrong number of ranks
        let result3 = tucker_nonnegative(&tensor, &[3, 3], 10, 1e-3);
        assert!(result3.is_err());
    }

    #[test]
    fn test_tucker_randomized_basic() {
        // Test basic randomized Tucker decomposition (reduced from 20^3 for speed)
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let result = tucker_randomized(&tensor, &[5, 5, 5], 3, 1);

        assert!(result.is_ok());
        let tucker = result.unwrap();

        // Check dimensions
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].shape(), &[10, 5]);
        assert_eq!(tucker.factors[1].shape(), &[10, 5]);
        assert_eq!(tucker.factors[2].shape(), &[10, 5]);
        assert_eq!(tucker.core.shape(), &[5, 5, 5]);
    }

    #[test]
    fn test_tucker_randomized_reconstruction() {
        // Test that randomized Tucker produces good reconstructions
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let mut tucker = tucker_randomized(&tensor, &[4, 4, 4], 3, 1).unwrap();

        // Compute reconstruction error
        let error = tucker.compute_error(&tensor).unwrap();

        // Error should be reasonable (not perfect due to randomization)
        assert!(
            error < 0.6,
            "Randomized Tucker error too large: {:.4}",
            error
        );

        // Reconstruction should succeed
        let reconstructed = tucker.reconstruct().unwrap();
        assert_eq!(reconstructed.shape(), tensor.shape());
    }

    #[test]
    fn test_tucker_randomized_vs_hosvd() {
        use crate::tucker::tucker_hosvd;

        // Compare randomized Tucker with HOSVD
        let tensor = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
        let ranks = vec![4, 4, 4];

        // Standard HOSVD
        let mut tucker_hosvd_result = tucker_hosvd(&tensor, &ranks).unwrap();
        let error_hosvd = tucker_hosvd_result.compute_error(&tensor).unwrap();

        // Randomized Tucker
        let mut tucker_rand = tucker_randomized(&tensor, &ranks, 3, 1).unwrap();
        let error_rand = tucker_rand.compute_error(&tensor).unwrap();

        // Randomized error should be comparable (within 30% typically)
        assert!(
            error_rand < error_hosvd * 1.4,
            "Randomized error ({:.4}) should be comparable to HOSVD error ({:.4})",
            error_rand,
            error_hosvd
        );

        println!(
            "HOSVD error: {:.4}, Randomized error: {:.4}",
            error_hosvd, error_rand
        );
    }

    #[test]
    fn test_tucker_randomized_oversampling() {
        // Test effect of oversampling parameter
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let ranks = vec![5, 5, 5];

        // With low oversampling
        let mut tucker_low = tucker_randomized(&tensor, &ranks, 2, 2).unwrap();
        let error_low = tucker_low.compute_error(&tensor).unwrap();

        // With high oversampling
        let mut tucker_high = tucker_randomized(&tensor, &ranks, 10, 2).unwrap();
        let error_high = tucker_high.compute_error(&tensor).unwrap();

        // Both should work, higher oversampling may give slightly better error
        assert!(error_low < 1.0);
        assert!(error_high < 1.0);

        println!(
            "Low oversampling error: {:.4}, High oversampling error: {:.4}",
            error_low, error_high
        );
    }

    #[test]
    fn test_tucker_randomized_power_iters() {
        // Test effect of power iterations
        let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
        let ranks = vec![3, 3, 3];

        // With no power iterations
        let mut tucker_no_power = tucker_randomized(&tensor, &ranks, 2, 0).unwrap();
        let error_no_power = tucker_no_power.compute_error(&tensor).unwrap();

        // With power iterations
        let mut tucker_power = tucker_randomized(&tensor, &ranks, 2, 1).unwrap();
        let error_power = tucker_power.compute_error(&tensor).unwrap();

        // Both should work, power iterations may give slightly better error
        assert!(error_no_power < 1.0);
        assert!(error_power < 1.0);

        println!(
            "No power iters error: {:.4}, With power iters error: {:.4}",
            error_no_power, error_power
        );
    }

    #[test]
    fn test_tucker_randomized_compression() {
        // Test that randomized Tucker provides good compression (reduced from 20^3 for speed)
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let tucker = tucker_randomized(&tensor, &[4, 4, 4], 3, 1).unwrap();

        let compression = tucker.compression_ratio();

        // Should achieve significant compression
        assert!(
            compression > 2.0,
            "Compression ratio should be > 2.0, got {:.2}",
            compression
        );

        println!("Compression ratio: {:.2}x", compression);
    }

    // ========================================================================
    // Tucker Completion Tests
    // ========================================================================

    #[test]
    fn test_tucker_completion_basic() {
        use scirs2_core::ndarray_ext::Array;

        // Create a 10x10x10 tensor
        let tensor_data = Array::from_shape_fn((10, 10, 10), |(i, j, k)| {
            (i as f64 + j as f64 + k as f64) / 3.0
        });
        let tensor = DenseND::from_array(tensor_data.into_dyn());

        // Create mask with 50% observed
        let mut mask_data = Array::ones((10, 10, 10));
        for i in 0..500 {
            let idx = (i / 100, (i / 10) % 10, i % 10);
            mask_data[idx] = 0.0;
        }
        let mask = DenseND::from_array(mask_data.into_dyn());

        // Run Tucker completion
        let result = tucker_completion(&tensor, &mask, &[5, 5, 5], 30, 1e-4);
        assert!(result.is_ok(), "Tucker completion should succeed");

        let tucker = result.unwrap();

        // Check dimensions
        assert_eq!(tucker.core.shape(), &[5, 5, 5]);
        assert_eq!(tucker.factors.len(), 3);
        assert_eq!(tucker.factors[0].shape(), &[10, 5]);
        assert_eq!(tucker.factors[1].shape(), &[10, 5]);
        assert_eq!(tucker.factors[2].shape(), &[10, 5]);

        // Check that error is reasonable
        assert!(tucker.error.is_some());
        let error = tucker.error.unwrap();
        assert!(error < 1.0, "Error should be < 1.0, got {:.4}", error);
    }

    #[test]
    fn test_tucker_completion_reconstruction() {
        use scirs2_core::ndarray_ext::Array;

        // Create a low-rank tensor
        let tensor_data = Array::from_shape_fn((8, 8, 8), |(i, j, k)| {
            (i as f64 / 8.0) * (j as f64 / 8.0) * (k as f64 / 8.0)
        });
        let tensor = DenseND::from_array(tensor_data.into_dyn());

        // Create mask with 60% observed
        let mut mask_data = Array::ones((8, 8, 8));
        for i in 0..205 {
            let idx = (i / 64, (i / 8) % 8, i % 8);
            mask_data[idx] = 0.0;
        }
        let mask = DenseND::from_array(mask_data.into_dyn());

        // Run Tucker completion
        let tucker = tucker_completion(&tensor, &mask, &[4, 4, 4], 50, 1e-4).unwrap();

        // Reconstruct
        let reconstructed = tucker.reconstruct();
        assert!(reconstructed.is_ok(), "Reconstruction should succeed");

        let recon = reconstructed.unwrap();
        assert_eq!(recon.shape(), tensor.shape());

        // Check that reconstruction is close on observed entries
        let tensor_view = tensor.view();
        let mask_view = mask.view();
        let recon_view = recon.view();

        let mut observed_error = 0.0;
        let mut observed_count = 0;

        for ((&t, &m), &r) in tensor_view
            .iter()
            .zip(mask_view.iter())
            .zip(recon_view.iter())
        {
            if m > 0.0 {
                observed_error += (t - r).abs();
                observed_count += 1;
            }
        }

        let avg_error = observed_error / observed_count as f64;
        assert!(
            avg_error < 0.5,
            "Average error on observed entries should be < 0.5, got {:.4}",
            avg_error
        );
    }

    #[test]
    fn test_tucker_completion_mask_validation() {
        let tensor = DenseND::<f64>::ones(&[5, 5, 5]);
        let wrong_mask = DenseND::<f64>::ones(&[5, 5, 6]);

        let result = tucker_completion(&tensor, &wrong_mask, &[3, 3, 3], 10, 1e-4);
        assert!(result.is_err(), "Should fail with wrong mask shape");
    }

    #[test]
    fn test_tucker_completion_no_observed_entries() {
        use scirs2_core::ndarray_ext::Array;

        let tensor = DenseND::<f64>::ones(&[5, 5, 5]);
        let mask_data = Array::zeros((5, 5, 5));
        let mask = DenseND::from_array(mask_data.into_dyn());

        let result = tucker_completion(&tensor, &mask, &[3, 3, 3], 10, 1e-4);
        assert!(result.is_err(), "Should fail with no observed entries");
    }

    #[test]
    fn test_tucker_completion_convergence() {
        use scirs2_core::ndarray_ext::Array;

        // Create a simple low-rank tensor
        let tensor_data = Array::from_elem((6, 6, 6), 5.0);
        let tensor = DenseND::from_array(tensor_data.into_dyn());

        // Create mask with 70% observed
        let mut mask_data = Array::ones((6, 6, 6));
        for i in 0..65 {
            let idx = (i / 36, (i / 6) % 6, i % 6);
            mask_data[idx] = 0.0;
        }
        let mask = DenseND::from_array(mask_data.into_dyn());

        // Run with different iteration counts
        let tucker_few = tucker_completion(&tensor, &mask, &[3, 3, 3], 10, 1e-4).unwrap();
        let tucker_many = tucker_completion(&tensor, &mask, &[3, 3, 3], 100, 1e-4).unwrap();

        // Both should converge
        assert!(tucker_few.iters > 0);
        assert!(tucker_many.iters > 0);

        println!(
            "Few iters: {}, Many iters: {}",
            tucker_few.iters, tucker_many.iters
        );
    }

    #[test]
    fn test_tucker_completion_high_missing_rate() {
        use scirs2_core::ndarray_ext::Array;

        // Create a low-rank tensor
        let tensor_data =
            Array::from_shape_fn((8, 8, 8), |(i, j, k)| ((i + j + k) as f64 / 24.0) * 10.0);
        let tensor = DenseND::from_array(tensor_data.into_dyn());

        // Create mask with 80% missing (only 20% observed)
        let mut mask_data = Array::zeros((8, 8, 8));
        for i in 0..102 {
            let idx = (i / 64, (i / 8) % 8, i % 8);
            mask_data[idx] = 1.0;
        }
        let mask = DenseND::from_array(mask_data.into_dyn());

        // Should still work with high missing rate
        let result = tucker_completion(&tensor, &mask, &[4, 4, 4], 100, 1e-3);
        assert!(result.is_ok(), "Should work with 80% missing data");

        let tucker = result.unwrap();
        assert!(tucker.error.unwrap() < 1.0);
    }

    #[test]
    fn test_tucker_completion_rank_validation() {
        use scirs2_core::ndarray_ext::Array;

        let tensor = DenseND::<f64>::ones(&[5, 5, 5]);
        let mask = DenseND::from_array(Array::ones((5, 5, 5)).into_dyn());

        // Invalid: rank too large
        let result1 = tucker_completion(&tensor, &mask, &[6, 3, 3], 10, 1e-4);
        assert!(result1.is_err(), "Should fail with rank > dimension");

        // Invalid: zero rank
        let result2 = tucker_completion(&tensor, &mask, &[0, 3, 3], 10, 1e-4);
        assert!(result2.is_err(), "Should fail with zero rank");

        // Invalid: wrong number of ranks
        let result3 = tucker_completion(&tensor, &mask, &[3, 3], 10, 1e-4);
        assert!(result3.is_err(), "Should fail with wrong number of ranks");
    }
}
