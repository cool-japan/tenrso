//! Internal helper functions for CP decomposition
//!
//! Contains compute kernels, initialization, and linear algebra utilities.

use super::types::*;
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView1};
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast, ToPrimitive};
use scirs2_core::random::{thread_rng, Distribution, RandNormal as Normal};
// lstsq replaced by inv for batch solve — see solve_least_squares
use std::iter::Sum;
use tenrso_core::DenseND;

/// Infallibly cast a literal / well-bounded numeric value to `T`.
///
/// SAFETY: all call sites use numeric literals or non-negative `usize`
/// values that are representable in any supported `T: Float` (f32, f64).
/// Returning `T::zero()` in the unreachable failure path preserves
/// numerical safety without a panic and respects the no-unwrap policy.
#[inline]
pub(crate) fn cast_lit<T: NumCast, V: ToPrimitive>(v: V) -> T {
    // SAFETY: `T: NumCast` is bounded by `Float` at all call sites
    // (f32/f64 at minimum), and `V` is a numeric literal or well-bounded
    // integer, so the conversion is infallible. In the unreachable failure
    // path we fall back to a zero-valued `T` via `T::from(0u8)`, which is
    // guaranteed by `NumCast` for primitive numeric types.
    T::from(v).unwrap_or_else(|| {
        T::from(0u8).unwrap_or_else(|| {
            unreachable!("NumCast::from(0u8) must succeed for primitive numeric T")
        })
    })
}

/// Create a standard normal distribution with fixed parameters.
///
/// SAFETY: `(mean, std_dev)` pairs used in this crate are literal
/// constants (e.g. `(0.0, 1.0)`, `(0.0, 0.01)`) that satisfy the
/// `Normal::new` precondition `std_dev > 0`. We surface the fallible
/// constructor through `Result` so callers can propagate via `?` and
/// avoid panicking unwraps.
#[inline]
pub(crate) fn make_normal(mean: f64, std_dev: f64) -> Result<Normal<f64>, CpError> {
    Normal::new(mean, std_dev).map_err(|e| {
        CpError::ShapeMismatch(format!(
            "failed to construct Normal({mean}, {std_dev}): {e}"
        ))
    })
}

/// Convert an `f64` tolerance/regularization scalar to `T` with a typed error.
///
/// SAFETY net: all call sites pass user-facing `f64` values (`tol`, `lambda`, ...)
/// that are always representable in the `T: Float` types supported by this crate.
#[inline]
pub(crate) fn cast_f64<T: NumCast>(val: f64, ctx: &'static str) -> Result<T, CpError> {
    NumCast::from(val).ok_or_else(|| {
        CpError::ShapeMismatch(format!("could not convert {ctx}={val} to scalar type"))
    })
}

/// Concatenate two tensors along a specified mode
pub(crate) fn concatenate_tensors<T>(
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

/// Reconstruct tensor from factors (for internal use)
pub(crate) fn tensor_from_factors<T>(
    factors: &[Array2<T>],
    weights: Option<&scirs2_core::ndarray_ext::Array1<T>>,
) -> Result<DenseND<T>>
where
    T: Float + NumCast + scirs2_core::ndarray_ext::ScalarOperand + 'static,
{
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let weights_view = weights.map(|w| w.view());

    let reconstructed = tenrso_kernels::cp_reconstruct(&factor_views, weights_view.as_ref())?;
    Ok(DenseND::from_array(reconstructed))
}

/// Compute Khatri-Rao product of all factors except one mode
pub(crate) fn compute_khatri_rao_except<T>(factors: &[Array2<T>], skip_mode: usize) -> Array2<T>
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

/// Compute tensor reconstruction from factors
pub(crate) fn compute_reconstruction<T>(factors: &[Array2<T>]) -> Result<DenseND<T>>
where
    T: Float + NumCast + scirs2_core::ndarray_ext::ScalarOperand + 'static,
{
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

    let reconstructed = tenrso_kernels::cp_reconstruct(&factor_views, None)?;
    Ok(DenseND::from_array(reconstructed))
}

// NOTE: a *per-mode* line search along the ALS direction used to live here. It was
// deleted, not moved, because it is provably a constant function.
//
// With every other factor held fixed, the CP objective restricted to factor `k` is
// the exact quadratic
//
//     f(A) = ‖X‖² − 2·⟨M_k, A⟩ + ⟨G_k, AᵀA⟩,     ∇f = 0  ⟺  A = M_k·G_k⁻¹,
//
// and `M_k·G_k⁻¹` is *precisely* what [`solve_least_squares`] returns. So the ALS
// step already lands on the global minimiser of the line it is searched along, and
// `argmin_α f(A_prev + α·(A_new − A_prev))` is identically `α = 1` — for every mode,
// every iteration, every tensor. A search that can only ever return 1 is not a line
// search.
//
// A line search only has something to find once the *whole factor set* moves at once.
// That is the ELS / extrapolation direction, and it lives in [`super::els`].

/// Initialize factor matrices based on strategy
pub(crate) fn initialize_factors<T>(
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
            for &mode_size in shape.iter() {
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    cast_lit::<T, _>(rng.random::<f64>())
                });
                factors.push(factor);
            }
        }
        InitStrategy::RandomNormal => {
            for &mode_size in shape.iter() {
                let normal = make_normal(0.0, 1.0)?;
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    cast_lit::<T, _>(normal.sample(&mut rng))
                });
                factors.push(factor);
            }
        }
        InitStrategy::Svd => {
            use scirs2_linalg::svd;

            for (mode, &mode_size) in shape.iter().enumerate() {
                let unfolded = tensor
                    .unfold(mode)
                    .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

                let (u, _s, _vt) =
                    svd(&unfolded.view(), false, None).map_err(CpError::LinalgError)?;

                let actual_rank = rank.min(u.shape()[1]);
                let mut factor = Array2::<T>::zeros((mode_size, rank));

                for i in 0..mode_size {
                    for j in 0..actual_rank {
                        factor[[i, j]] = u[[i, j]];
                    }
                }

                if rank > actual_rank {
                    let normal = make_normal(0.0, 0.01)?;
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            factor[[i, j]] = cast_lit::<T, _>(normal.sample(&mut rng));
                        }
                    }
                }

                factors.push(factor);
            }
        }
        InitStrategy::Nnsvd => {
            use scirs2_linalg::svd;

            for (mode, &mode_size) in shape.iter().enumerate() {
                let unfolded = tensor
                    .unfold(mode)
                    .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

                let (u, s, vt) =
                    svd(&unfolded.view(), false, None).map_err(CpError::LinalgError)?;

                let actual_rank = rank.min(u.shape()[1]).min(vt.shape()[0]);
                let mut factor = Array2::<T>::zeros((mode_size, rank));

                for r in 0..actual_rank {
                    let u_col = u.column(r);
                    let v_row = vt.row(r);

                    let (u_pos, u_neg) = split_sign(&u_col);
                    let (v_pos, v_neg) = split_sign(&v_row);

                    let u_pos_norm = compute_vec_norm(&u_pos);
                    let u_neg_norm = compute_vec_norm(&u_neg);
                    let v_pos_norm = compute_vec_norm(&v_pos);
                    let v_neg_norm = compute_vec_norm(&v_neg);

                    let pos_prod = u_pos_norm * v_pos_norm;
                    let neg_prod = u_neg_norm * v_neg_norm;

                    if pos_prod >= neg_prod {
                        let scale = (s[r] * pos_prod).sqrt();
                        for i in 0..mode_size {
                            factor[[i, r]] = u_pos[i] * scale;
                        }
                    } else {
                        let scale = (s[r] * neg_prod).sqrt();
                        for i in 0..mode_size {
                            factor[[i, r]] = u_neg[i] * scale;
                        }
                    }
                }

                if rank > actual_rank {
                    let normal = make_normal(0.0, 0.01)?;
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            let val = cast_lit::<T, _>(normal.sample(&mut rng));
                            factor[[i, j]] = val.abs();
                        }
                    }
                }

                factors.push(factor);
            }
        }
        InitStrategy::LeverageScore => {
            use scirs2_linalg::svd;

            for (mode, &mode_size) in shape.iter().enumerate() {
                let unfolded = tensor
                    .unfold(mode)
                    .map_err(|e| CpError::ShapeMismatch(format!("Unfold failed: {}", e)))?;

                let (u, s, _vt) =
                    svd(&unfolded.view(), false, None).map_err(CpError::LinalgError)?;

                let actual_rank = rank.min(u.shape()[1]).min(s.len());

                let mut leverage_scores = vec![T::zero(); mode_size];
                // SAFETY: actual_rank is always a positive usize (bounded by SVD dims);
                // representable in f32/f64, the crate's supported `T: Float` types.
                let actual_rank_t = T::from(actual_rank).ok_or_else(|| {
                    CpError::ShapeMismatch(format!(
                        "could not convert actual_rank={actual_rank} to scalar type",
                    ))
                })?;
                for i in 0..mode_size {
                    let mut score = T::zero();
                    for j in 0..actual_rank {
                        let val = u[[i, j]];
                        score += val * val;
                    }
                    leverage_scores[i] = score / actual_rank_t;
                }

                let total_score: T = leverage_scores.iter().copied().sum();
                if total_score > T::epsilon() {
                    for score in &mut leverage_scores {
                        *score /= total_score;
                    }
                }

                let mut factor = Array2::<T>::zeros((mode_size, rank));

                // SAFETY: mode_size is always a positive usize (tensor dimension);
                // representable in any supported `T: Float`.
                let mode_size_t = T::from(mode_size).ok_or_else(|| {
                    CpError::ShapeMismatch(format!(
                        "could not convert mode_size={mode_size} to scalar type",
                    ))
                })?;
                for r in 0..actual_rank {
                    let weight = s[r].sqrt();
                    for i in 0..mode_size {
                        let leverage_weight = (leverage_scores[i] * mode_size_t).sqrt();
                        factor[[i, r]] = u[[i, r]] * weight * leverage_weight;
                    }
                }

                if rank > actual_rank {
                    let normal = make_normal(0.0, 0.01)?;
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            let base_val = cast_lit::<T, _>(normal.sample(&mut rng));
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
pub(crate) fn split_sign<T>(vec: &ArrayView1<T>) -> (Vec<T>, Vec<T>)
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
            neg.push(-val);
        }
    }

    (pos, neg)
}

/// Compute L2 norm of a vector
pub(crate) fn compute_vec_norm<T>(vec: &[T]) -> T
where
    T: Float + Sum,
{
    vec.iter().map(|&x| x * x).sum::<T>().sqrt()
}

/// Compute Hadamard product of Gram matrices for all factors except one mode
pub(crate) fn compute_gram_hadamard<T>(factors: &[Array2<T>], skip_mode: usize) -> Array2<T>
where
    T: Float,
{
    let rank = factors[0].shape()[1];
    let mut gram = Array2::<T>::ones((rank, rank));

    for (i, factor) in factors.iter().enumerate() {
        if i == skip_mode {
            continue;
        }

        let factor_gram = compute_gram_matrix(factor);

        for r1 in 0..rank {
            for r2 in 0..rank {
                gram[[r1, r2]] = gram[[r1, r2]] * factor_gram[[r1, r2]];
            }
        }
    }

    gram
}

/// Compute Gram matrix: F^T F
pub(crate) fn compute_gram_matrix<T>(factor: &Array2<T>) -> Array2<T>
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

/// Solve least squares problem: X = MTTKRP × gram⁻¹
///
/// Instead of solving per-row via `lstsq` (which repeats O(R³) work for each
/// of the I_mode rows), we invert the small R×R Gram matrix *once* and multiply.
/// This reduces the solve from O(I_mode × R³) to O(R³ + I_mode × R²).
pub(crate) fn solve_least_squares<T>(
    mttkrp_result: &Array2<T>,
    gram: &Array2<T>,
) -> Result<Array2<T>, CpError>
where
    T: Float + NumAssign + Sum + scirs2_core::ndarray_ext::ScalarOperand + Send + Sync + 'static,
{
    use scirs2_linalg::inv;

    // Try to invert the R×R Gram matrix directly (very cheap for R=64).
    let gram_inv = match inv(&gram.view(), None) {
        Ok(g_inv) => g_inv,
        Err(_) => {
            // Gram is singular/ill-conditioned — add small ridge and retry.
            let rank = gram.shape()[0];
            let eps = T::epsilon() * cast_lit::<T, _>(rank * 10);
            let mut gram_reg = gram.clone();
            for k in 0..rank {
                gram_reg[[k, k]] += eps;
            }
            inv(&gram_reg.view(), None).map_err(CpError::LinalgError)?
        }
    };

    // Batch multiply: result = MTTKRP × gram_inv^T
    // Each row i: result[i,:] = mttkrp_result[i,:] · gram_inv^T
    // Equivalently: result = mttkrp_result · gram_inv^T
    // Note: for symmetric Gram, gram_inv is also symmetric, so gram_inv^T = gram_inv.
    // We transpose to match the original lstsq convention (gram^T was passed).
    let result = mttkrp_result.dot(&gram_inv.t());

    Ok(result)
}

/// Compute squared Frobenius norm of tensor
pub(crate) fn compute_norm_squared<T>(tensor: &DenseND<T>) -> T
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
pub(crate) fn compute_fit<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
    tensor_norm_sq: T,
) -> Result<T, CpError>
where
    T: Float + NumCast + 'static,
{
    let recon_norm_sq = compute_reconstruction_norm_squared(factors);
    let inner_product = compute_inner_product(tensor, factors)?;

    let error_sq = tensor_norm_sq + recon_norm_sq - cast_lit::<T, _>(2) * inner_product;
    let error = error_sq.max(T::zero()).sqrt();

    let fit = T::one() - error / tensor_norm_sq.sqrt();

    Ok(fit.max(T::zero()).min(T::one()))
}

/// Compute the fit from a MTTKRP that is already consistent with `factors`.
///
/// `mttkrp_mode` must be a mode index and `mttkrp` the corresponding MTTKRP taken
/// against *the very factors passed here* (i.e. `factors[j]` for all `j != mode`).
/// Under that condition
///
/// ```text
/// <X, [[A]]> = sum_{i, r} mttkrp[i, r] * factors[mode][i, r]
/// ```
///
/// holds for **every** mode — the inner product does not care which mode you route
/// it through. A Gauss-Seidel ALS sweep updates mode `N-1` last, so the sweep's
/// final MTTKRP already satisfies the condition for `mode = N-1` and the fit costs
/// no extra tensor pass.
///
/// This is the identical formula to [`compute_fit`] (which routes the inner product
/// through a freshly recomputed mode-0 MTTKRP); the two agree to floating-point
/// tolerance. See `cp::tests::cp_als_fit_reuse_matches_recomputed_fit`.
pub(crate) fn compute_fit_from_mttkrp<T>(
    factors: &[Array2<T>],
    mttkrp: &Array2<T>,
    mttkrp_mode: usize,
    tensor_norm_sq: T,
) -> T
where
    T: Float + NumCast + 'static,
{
    let recon_norm_sq = compute_reconstruction_norm_squared(factors);
    let inner_product = compute_inner_product_from_mttkrp(mttkrp, &factors[mttkrp_mode]);

    let error_sq = tensor_norm_sq + recon_norm_sq - cast_lit::<T, _>(2) * inner_product;
    let error = error_sq.max(T::zero()).sqrt();

    let fit = T::one() - error / tensor_norm_sq.sqrt();

    fit.max(T::zero()).min(T::one())
}

/// Compute ||X_recon||^2 from factor matrices
pub(crate) fn compute_reconstruction_norm_squared<T>(factors: &[Array2<T>]) -> T
where
    T: Float,
{
    let rank = factors[0].shape()[1];
    let mut norm_sq = T::zero();

    for r in 0..rank {
        for s in 0..rank {
            let mut cross_term = T::one();
            for factor in factors {
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

/// Compute inner product <X, X_recon>
pub(crate) fn compute_inner_product<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
) -> Result<T, CpError>
where
    T: Float + NumCast + 'static,
{
    use tenrso_kernels::mttkrp;
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let mttkrp_result = mttkrp(&tensor.view(), &factor_views, 0)
        .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

    Ok(compute_inner_product_from_mttkrp(
        &mttkrp_result,
        &factors[0],
    ))
}

/// Compute inner product <X, X_recon> from a pre-computed MTTKRP result.
///
/// This avoids a redundant MTTKRP computation when the caller already has
/// the MTTKRP for mode 0 available (e.g., from the last ALS factor update).
pub(crate) fn compute_inner_product_from_mttkrp<T>(
    mttkrp_result: &Array2<T>,
    factor_mode0: &Array2<T>,
) -> T
where
    T: Float,
{
    let rank = factor_mode0.shape()[1];
    let mut inner_prod = T::zero();

    for r in 0..rank {
        for i in 0..factor_mode0.shape()[0] {
            inner_prod = inner_prod + mttkrp_result[[i, r]] * factor_mode0[[i, r]];
        }
    }

    inner_prod
}

/// Orthonormalize a factor matrix using QR decomposition
pub(crate) fn orthonormalize_factor<T>(factor: &Array2<T>) -> Result<Array2<T>, CpError>
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

    let (q_full, _r) = qr(&factor.view(), None).map_err(CpError::LinalgError)?;

    let q = q_full.slice(s![.., ..n]).to_owned();

    Ok(q)
}

/// Apply soft-thresholding (proximal operator for L1 regularization)
///
/// For each element: sign(x) * max(|x| - threshold, 0)
pub(crate) fn soft_threshold<T>(factor: &mut Array2<T>, threshold: T)
where
    T: Float,
{
    factor.mapv_inplace(|x| {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            T::zero()
        }
    });
}

/// Apply Tikhonov regularization to the Gram matrix
///
/// For order 0: Adds lambda * I (standard ridge)
/// For order 1: Adds lambda * D1^T * D1 where D1 is first-order finite difference
/// For order 2: Adds lambda * D2^T * D2 where D2 is second-order finite difference
pub(crate) fn apply_tikhonov_to_gram<T>(gram: &mut Array2<T>, lambda: T, order: usize)
where
    T: Float + NumCast,
{
    let n = gram.shape()[0];

    match order {
        0 => {
            // Standard ridge: add lambda * I
            for i in 0..n {
                gram[[i, i]] = gram[[i, i]] + lambda;
            }
        }
        1 => {
            // First-order finite difference: D1^T * D1
            // D1 is (n-1) x n with D1[i,i] = -1, D1[i,i+1] = 1
            // D1^T * D1 has:
            //   diagonal: 1 at corners, 2 elsewhere
            //   off-diagonal (+-1): -1
            if n >= 2 {
                for i in 0..n {
                    let diag_val = if i == 0 || i == n - 1 {
                        T::one()
                    } else {
                        cast_lit::<T, _>(2)
                    };
                    gram[[i, i]] = gram[[i, i]] + lambda * diag_val;

                    if i + 1 < n {
                        gram[[i, i + 1]] = gram[[i, i + 1]] - lambda;
                        gram[[i + 1, i]] = gram[[i + 1, i]] - lambda;
                    }
                }
            }
        }
        _ => {
            // Higher orders: fall back to ridge for simplicity
            // Second-order is D2^T * D2 where D2[i,:] = [1, -2, 1]
            if n >= 3 {
                for i in 0..n {
                    let diag_val = if i == 0 || i == n - 1 {
                        T::one()
                    } else if i == 1 || i == n - 2 {
                        cast_lit::<T, _>(5)
                    } else {
                        cast_lit::<T, _>(6)
                    };
                    gram[[i, i]] = gram[[i, i]] + lambda * diag_val;

                    if i + 1 < n {
                        let off1 = if i == 0 || i == n - 2 {
                            cast_lit::<T, _>(-2)
                        } else {
                            cast_lit::<T, _>(-4)
                        };
                        gram[[i, i + 1]] = gram[[i, i + 1]] + lambda * off1;
                        gram[[i + 1, i]] = gram[[i + 1, i]] + lambda * off1;
                    }

                    if i + 2 < n {
                        gram[[i, i + 2]] = gram[[i, i + 2]] + lambda;
                        gram[[i + 2, i]] = gram[[i + 2, i]] + lambda;
                    }
                }
            } else {
                // Too small for second-order, use ridge
                for i in 0..n {
                    gram[[i, i]] = gram[[i, i]] + lambda;
                }
            }
        }
    }
}

/// Validate that a tensor contains only non-negative values
pub(crate) fn validate_nonnegative<T>(tensor: &DenseND<T>) -> Result<(), CpError>
where
    T: Float,
{
    let view = tensor.view();
    for &val in view.iter() {
        if val < T::zero() {
            return Err(CpError::NonnegativeViolation);
        }
    }
    Ok(())
}
