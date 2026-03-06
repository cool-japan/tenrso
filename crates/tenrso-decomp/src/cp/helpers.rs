//! Internal helper functions for CP decomposition
//!
//! Contains compute kernels, initialization, and linear algebra utilities.

use super::types::*;
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView1};
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use scirs2_core::random::{thread_rng, Distribution, RandNormal as Normal};
use scirs2_linalg::lstsq;
use std::iter::Sum;
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;

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

/// Line search to find optimal step size for CP-ALS update
pub(crate) fn line_search_cp<T>(
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
    let tensor_norm_sq = tensor_norm * tensor_norm;
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
                    T::from(rng.random::<f64>()).unwrap()
                });
                factors.push(factor);
            }
        }
        InitStrategy::RandomNormal => {
            for &mode_size in shape.iter() {
                let normal = Normal::new(0.0, 1.0).unwrap();
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    T::from(normal.sample(&mut rng)).unwrap()
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
                    let normal = Normal::new(0.0, 0.01).unwrap();
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
                            let val = T::from(normal.sample(&mut rng)).unwrap();
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
                for i in 0..mode_size {
                    let mut score = T::zero();
                    for j in 0..actual_rank {
                        let val = u[[i, j]];
                        score += val * val;
                    }
                    leverage_scores[i] = score / T::from(actual_rank).unwrap();
                }

                let total_score: T = leverage_scores.iter().copied().sum();
                if total_score > T::epsilon() {
                    for score in &mut leverage_scores {
                        *score /= total_score;
                    }
                }

                let mut factor = Array2::<T>::zeros((mode_size, rank));

                for r in 0..actual_rank {
                    let weight = s[r].sqrt();
                    for i in 0..mode_size {
                        let leverage_weight =
                            (leverage_scores[i] * T::from(mode_size).unwrap()).sqrt();
                        factor[[i, r]] = u[[i, r]] * weight * leverage_weight;
                    }
                }

                if rank > actual_rank {
                    let normal = Normal::new(0.0, 0.01).unwrap();
                    for j in actual_rank..rank {
                        for i in 0..mode_size {
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

/// Solve least squares problem: X = A * gram^(-1)
pub(crate) fn solve_least_squares<T>(
    mttkrp_result: &Array2<T>,
    gram: &Array2<T>,
) -> Result<Array2<T>, CpError>
where
    T: Float + NumAssign + Sum + scirs2_core::ndarray_ext::ScalarOperand + Send + Sync + 'static,
{
    let (rows, rank) = (mttkrp_result.shape()[0], mttkrp_result.shape()[1]);

    let gram_t = gram.t().to_owned();

    let mut result = Array2::<T>::zeros((rows, rank));

    for i in 0..rows {
        let b = mttkrp_result.row(i).to_owned();

        match lstsq(&gram_t.view(), &b.view(), None) {
            Ok(solution) => {
                for j in 0..rank {
                    result[[i, j]] = solution.x[j];
                }
            }
            Err(_) => {
                let eps = T::epsilon() * T::from(rank * 10).unwrap();
                let mut gram_reg = gram_t.clone();
                for k in 0..rank.min(gram_reg.shape()[0]) {
                    gram_reg[[k, k]] += eps;
                }

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
    T: Float + NumCast,
{
    let recon_norm_sq = compute_reconstruction_norm_squared(factors);
    let inner_product = compute_inner_product(tensor, factors)?;

    let error_sq = tensor_norm_sq + recon_norm_sq - T::from(2).unwrap() * inner_product;
    let error = error_sq.max(T::zero()).sqrt();

    let fit = T::one() - error / tensor_norm_sq.sqrt();

    Ok(fit.max(T::zero()).min(T::one()))
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
    T: Float,
{
    let mut inner_prod = T::zero();
    let rank = factors[0].shape()[1];

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
                        T::from(2).unwrap()
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
                        T::from(5).unwrap()
                    } else {
                        T::from(6).unwrap()
                    };
                    gram[[i, i]] = gram[[i, i]] + lambda * diag_val;

                    if i + 1 < n {
                        let off1 = if i == 0 || i == n - 2 {
                            T::from(-2).unwrap()
                        } else {
                            T::from(-4).unwrap()
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
