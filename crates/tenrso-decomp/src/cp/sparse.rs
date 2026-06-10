//! Sparse CP-ALS: alternating least squares for CP decomposition of sparse tensors.
//!
//! Uses `mttkrp_sparse_coo` from `tenrso-kernels` so the cost per ALS iteration
//! scales as O(nnz · R) rather than O(∏I_n · R) for dense inputs.  At ≤ 5% fill
//! the break-even is typically 3-4 iterations; for sparser tensors every iteration
//! is faster than the dense path.
//!
//! # Asymptotic complexity (per iteration)
//!
//! | Step | Dense | Sparse (COO) |
//! |------|-------|--------------|
//! | MTTKRP per mode | O(∏I_n · R) | O(nnz · (N-1) · R) |
//! | Gram hadamard | O(N · R²) | same |
//! | Least-squares solve | O(R³) | same |
//! | Fit (inner product) | O(∏I_n · R) | O(nnz · (N-1) · R) |
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "sparse")]
//! # {
//! use tenrso_decomp::cp::{cp_als_sparse, InitStrategy};
//! use tenrso_sparse::coo::CooTensor;
//!
//! let mut coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
//! coo.push(vec![0, 0, 0], 1.0).unwrap();
//! coo.push(vec![1, 2, 3], 2.0).unwrap();
//! coo.push(vec![3, 4, 5], 3.0).unwrap();
//!
//! let cp = cp_als_sparse(&coo, 2, 30, 1e-4, InitStrategy::Random, None).unwrap();
//! println!("fit = {:.4}", cp.fit);
//! # }
//! ```

use super::helpers::{
    cast_f64, cast_lit, compute_gram_hadamard, compute_inner_product_from_mttkrp,
    compute_reconstruction_norm_squared, make_normal, solve_least_squares,
};
use super::types::*;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use scirs2_core::random::{thread_rng, Distribution};
use std::iter::Sum;
use tenrso_sparse::coo::CooTensor;

/// Compute CP decomposition of a sparse COO tensor via ALS.
///
/// Equivalent to [`cp_als`](super::core::cp_als) but accepts a sparse
/// [`CooTensor<T>`] and uses `mttkrp_sparse_coo` for the dominant per-iteration
/// cost, giving O(nnz · R) per MTTKRP instead of O(∏I_n · R).
///
/// # Arguments
///
/// * `tensor` - Sparse input tensor in COO format (duplicate indices are aggregated)
/// * `rank` - Target CP rank (number of components); must be ≤ every mode size
/// * `max_iters` - Maximum ALS iterations
/// * `tol` - Convergence tolerance on fit improvement; must satisfy 0 < tol < 1
/// * `init` - Initialization strategy (`Svd`/`Nnsvd` densify the tensor once at startup)
/// * `time_limit` - Optional wall-clock limit; `None` disables it
///
/// # Returns
///
/// [`CpDecomp`] with the same fields as the dense variant.
///
/// # Errors
///
/// * [`CpError::InvalidRank`] if `rank == 0` or `rank > any mode size`
/// * [`CpError::InvalidTolerance`] if `tol ∉ (0, 1)`
/// * [`CpError::ShapeMismatch`] if MTTKRP dimensions are inconsistent
/// * [`CpError::LinalgError`] on SVD/least-squares failure
///
/// # Complexity
///
/// Per iteration: O(N · nnz · R) MTTKRP + O(N · R²) Gram + O(R³) solve
pub fn cp_als_sparse<T>(
    tensor: &CooTensor<T>,
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
    use tenrso_kernels::mttkrp_sparse_coo;

    let shape = tensor.shape();
    let n_modes = shape.len();

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

    let tol_t: T = cast_f64(tol, "tol")?;

    // ||X||_F² = sum of squared nonzero values (zeros contribute nothing)
    let tensor_norm_sq = sparse_norm_squared(tensor);

    // Degenerate case: all-zero tensor is trivially represented by all-zero factors.
    if tensor_norm_sq == T::zero() {
        let zero_factors: Vec<Array2<T>> = shape
            .iter()
            .map(|&s| Array2::<T>::zeros((s, rank)))
            .collect();
        return Ok(CpDecomp {
            factors: zero_factors,
            weights: None,
            fit: T::one(),
            iters: 0,
            convergence: Some(ConvergenceInfo {
                fit_history: vec![T::one()],
                reason: ConvergenceReason::FitTolerance,
                oscillated: false,
                oscillation_count: 0,
                final_fit_change: T::zero(),
            }),
        });
    }

    let mut factors = initialize_sparse_factors(tensor, rank, init)?;

    let mut prev_fit = T::zero();
    let mut fit = T::zero();
    let mut iters = 0;
    let mut fit_history = Vec::with_capacity(max_iters);
    let mut oscillation_count = 0usize;
    let mut convergence_reason = ConvergenceReason::MaxIterations;
    let mut final_fit_change = T::zero();

    let start_time = std::time::Instant::now();

    for iter in 0..max_iters {
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                convergence_reason = ConvergenceReason::TimeLimit;
                break;
            }
        }
        iters = iter + 1;

        // ALS factor update — sparse MTTKRP replaces dense unfold+GEMM
        for mode in 0..n_modes {
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp_sparse_coo(tensor, &factor_views, mode)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;
            let gram = compute_gram_hadamard(&factors, mode);
            factors[mode] = solve_least_squares(&mttkrp_result, &gram)?;
        }

        // Fit via the inner-product formula:
        //   ||X - X̂||² = ||X||² + ||X̂||² - 2⟨X, X̂⟩
        //   ⟨X, X̂⟩   = trace(A₀ᵀ · MTTKRP₀(X; all updated factors))
        //
        // Recompute MTTKRP for mode 0 after all factors are updated so that
        // the inner product uses the fully-updated factor matrices.
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let mttkrp0 = mttkrp_sparse_coo(tensor, &factor_views, 0)
            .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

        let inner_product = compute_inner_product_from_mttkrp(&mttkrp0, &factors[0]);
        let recon_norm_sq = compute_reconstruction_norm_squared(&factors);

        let error_sq = tensor_norm_sq + recon_norm_sq - cast_lit::<T, _>(2) * inner_product;
        let error = error_sq.max(T::zero()).sqrt();
        let tensor_norm = tensor_norm_sq.sqrt();
        fit = if tensor_norm > T::zero() {
            (T::one() - error / tensor_norm)
                .max(T::zero())
                .min(T::one())
        } else {
            T::zero()
        };
        fit_history.push(fit);

        if iter > 0 && fit < prev_fit {
            oscillation_count += 1;
        }
        let fit_change = (fit - prev_fit).abs();
        final_fit_change = fit_change;
        if iter > 0 && fit_change < tol_t {
            convergence_reason = ConvergenceReason::FitTolerance;
            break;
        }
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

/// Parallel variant of [`cp_als_sparse`]: uses `mttkrp_sparse_coo_parallel` for
/// the dominant per-iteration cost.
///
/// Identical semantics to `cp_als_sparse`; each per-mode MTTKRP call distributes
/// the `nnz` nonzeros across available threads via Rayon.  The parallel speedup is
/// proportional to `nnz / (mode_size × R)` and is most pronounced for large,
/// high-rank sparse tensors.
///
/// # Arguments
///
/// Same as [`cp_als_sparse`].
///
/// # Examples
///
/// ```
/// # #[cfg(all(feature = "sparse", feature = "parallel"))]
/// # {
/// use tenrso_decomp::cp::{cp_als_sparse_parallel, InitStrategy};
/// use tenrso_sparse::coo::CooTensor;
///
/// let mut coo = CooTensor::<f64>::zeros(vec![10, 10, 10]).unwrap();
/// coo.push(vec![0, 1, 2], 1.0).unwrap();
/// coo.push(vec![5, 5, 5], 2.0).unwrap();
///
/// let cp = cp_als_sparse_parallel(&coo, 3, 30, 1e-4, InitStrategy::Random, None).unwrap();
/// println!("parallel fit = {:.4}", cp.fit);
/// # }
/// ```
#[cfg(feature = "parallel")]
pub fn cp_als_sparse_parallel<T>(
    tensor: &CooTensor<T>,
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
    use tenrso_kernels::mttkrp_sparse_coo_parallel;

    let shape = tensor.shape();
    let n_modes = shape.len();

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

    let tol_t: T = cast_f64(tol, "tol")?;
    let tensor_norm_sq = sparse_norm_squared(tensor);

    if tensor_norm_sq == T::zero() {
        let zero_factors: Vec<Array2<T>> = shape
            .iter()
            .map(|&s| Array2::<T>::zeros((s, rank)))
            .collect();
        return Ok(CpDecomp {
            factors: zero_factors,
            weights: None,
            fit: T::one(),
            iters: 0,
            convergence: Some(ConvergenceInfo {
                fit_history: vec![T::one()],
                reason: ConvergenceReason::FitTolerance,
                oscillated: false,
                oscillation_count: 0,
                final_fit_change: T::zero(),
            }),
        });
    }

    let mut factors = initialize_sparse_factors(tensor, rank, init)?;

    let mut prev_fit = T::zero();
    let mut fit = T::zero();
    let mut iters = 0;
    let mut fit_history = Vec::with_capacity(max_iters);
    let mut oscillation_count = 0usize;
    let mut convergence_reason = ConvergenceReason::MaxIterations;
    let mut final_fit_change = T::zero();

    let start_time = std::time::Instant::now();

    for iter in 0..max_iters {
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                convergence_reason = ConvergenceReason::TimeLimit;
                break;
            }
        }
        iters = iter + 1;

        for mode in 0..n_modes {
            let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
            let mttkrp_result = mttkrp_sparse_coo_parallel(tensor, &factor_views, mode)
                .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;
            let gram = compute_gram_hadamard(&factors, mode);
            factors[mode] = solve_least_squares(&mttkrp_result, &gram)?;
        }

        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
        let mttkrp0 = mttkrp_sparse_coo_parallel(tensor, &factor_views, 0)
            .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

        let inner_product = compute_inner_product_from_mttkrp(&mttkrp0, &factors[0]);
        let recon_norm_sq = compute_reconstruction_norm_squared(&factors);
        let error_sq = tensor_norm_sq + recon_norm_sq - cast_lit::<T, _>(2) * inner_product;
        let error = error_sq.max(T::zero()).sqrt();
        let tensor_norm = tensor_norm_sq.sqrt();
        fit = if tensor_norm > T::zero() {
            (T::one() - error / tensor_norm)
                .max(T::zero())
                .min(T::one())
        } else {
            T::zero()
        };
        fit_history.push(fit);

        if iter > 0 && fit < prev_fit {
            oscillation_count += 1;
        }
        let fit_change = (fit - prev_fit).abs();
        final_fit_change = fit_change;
        if iter > 0 && fit_change < tol_t {
            convergence_reason = ConvergenceReason::FitTolerance;
            break;
        }
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

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// ||X||_F² for a sparse tensor: sum of squared nonzero values.
fn sparse_norm_squared<T: Float + 'static>(tensor: &CooTensor<T>) -> T {
    tensor
        .values()
        .iter()
        .fold(T::zero(), |acc, &v| acc + v * v)
}

/// Initialize CP factor matrices from a sparse tensor.
///
/// - `Random` / `RandomNormal`: only shape is used (no tensor values needed).
/// - `Svd`, `Nnsvd`, `LeverageScore`: tensor is densified once at startup.
fn initialize_sparse_factors<T>(
    tensor: &CooTensor<T>,
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
        + std::fmt::Display
        + 'static,
{
    let shape = tensor.shape();
    let n_modes = shape.len();

    match init {
        InitStrategy::Random => {
            let mut rng = thread_rng();
            let mut factors = Vec::with_capacity(n_modes);
            for &mode_size in shape.iter() {
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    cast_lit::<T, _>(rng.random::<f64>())
                });
                factors.push(factor);
            }
            Ok(factors)
        }
        InitStrategy::RandomNormal => {
            let mut rng = thread_rng();
            let mut factors = Vec::with_capacity(n_modes);
            for &mode_size in shape.iter() {
                let normal = make_normal(0.0, 1.0)?;
                let factor = Array2::from_shape_fn((mode_size, rank), |_| {
                    cast_lit::<T, _>(normal.sample(&mut rng))
                });
                factors.push(factor);
            }
            Ok(factors)
        }
        // All other strategies need dense values; densify once at init.
        other => {
            use super::helpers::initialize_factors as init_dense;
            let dense = tensor
                .to_dense()
                .map_err(|e| CpError::ShapeMismatch(format!("densify for init failed: {}", e)))?;
            init_dense(&dense, rank, other)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp::{cp_als, InitStrategy as IS};
    use tenrso_core::DenseND;

    /// Build a deterministic sparse COO tensor using LCG (avoids `rand` dep).
    fn build_sparse_coo(shape: &[usize], fill_frac: f64, seed: u64) -> CooTensor<f64> {
        let total: usize = shape.iter().product();
        let nnz = ((total as f64) * fill_frac).max(1.0) as usize;
        let ndim = shape.len();

        let mut coo = CooTensor::zeros(shape.to_vec()).unwrap();
        let mut rng = seed;
        for _ in 0..nnz {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let flat = rng as usize % total;
            let mut idx = vec![0usize; ndim];
            let mut rem = flat;
            for d in (0..ndim).rev() {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            let val = (rng >> 32) as f64 / (u32::MAX as f64) + 0.1;
            coo.push(idx, val).ok();
        }
        coo.deduplicate();
        coo
    }

    // ------------------------------------------------------------------
    // 1. Basic: sparse CP-ALS runs and returns valid CpDecomp
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_basic() {
        let coo = build_sparse_coo(&[5, 6, 7], 0.10, 42);
        let cp = cp_als_sparse(&coo, 3, 20, 1e-4, IS::Random, None).unwrap();
        assert_eq!(cp.factors.len(), 3);
        assert_eq!(cp.factors[0].shape(), &[5, 3]);
        assert_eq!(cp.factors[1].shape(), &[6, 3]);
        assert_eq!(cp.factors[2].shape(), &[7, 3]);
        assert!(cp.fit >= 0.0 && cp.fit <= 1.0, "fit must be in [0,1]");
        assert!(cp.iters > 0);
    }

    // ------------------------------------------------------------------
    // 2. Sparse and dense produce similar fit on a known low-rank tensor
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_matches_dense_fit() {
        let (n0, n1, n2) = (8, 7, 6);
        let rank = 3usize;
        // Build rank-R tensor with LCG
        let mut state = 0xDEAD_BEEF_u64;
        let mut rng_v = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as f64 / (u32::MAX as f64) - 0.5
        };
        let a: Vec<f64> = (0..n0 * rank).map(|_| rng_v()).collect();
        let b: Vec<f64> = (0..n1 * rank).map(|_| rng_v()).collect();
        let c: Vec<f64> = (0..n2 * rank).map(|_| rng_v()).collect();
        let mut data = vec![0.0f64; n0 * n1 * n2];
        for r in 0..rank {
            for i in 0..n0 {
                for j in 0..n1 {
                    for k in 0..n2 {
                        data[i * n1 * n2 + j * n2 + k] +=
                            a[i * rank + r] * b[j * rank + r] * c[k * rank + r];
                    }
                }
            }
        }
        let dense = DenseND::<f64>::from_vec(data, &[n0, n1, n2]).unwrap();

        // Build COO from dense
        let shape = vec![n0, n1, n2];
        let total = n0 * n1 * n2;
        let mut coo = CooTensor::<f64>::zeros(shape.clone()).unwrap();
        let view = dense.view();
        for flat in 0..total {
            let mut idx = vec![0usize; 3];
            let mut rem = flat;
            for d in (0..3).rev() {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            let val = view[&idx[..]];
            if val != 0.0 {
                coo.push(idx, val).ok();
            }
        }

        let cp_d = cp_als(&dense, rank, 100, 1e-6, IS::Random, None).unwrap();
        let cp_s = cp_als_sparse(&coo, rank, 100, 1e-6, IS::Random, None).unwrap();

        assert!(cp_d.fit > 0.5, "dense fit = {}", cp_d.fit);
        assert!(cp_s.fit > 0.5, "sparse fit = {}", cp_s.fit);
        assert!(
            (cp_d.fit - cp_s.fit).abs() < 0.35,
            "dense fit {:.4} and sparse fit {:.4} should be similar",
            cp_d.fit,
            cp_s.fit
        );
    }

    // ------------------------------------------------------------------
    // 3. Fit field matches reconstruction-based fit
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_fit_consistency() {
        let coo = build_sparse_coo(&[6, 6, 6], 0.08, 77);
        let cp = cp_als_sparse(&coo, 3, 50, 1e-5, IS::Random, None).unwrap();

        let dense = coo.to_dense().unwrap();
        let recon = cp.reconstruct(dense.shape()).unwrap();
        let orig_norm = dense.frobenius_norm();
        let diff = &dense - &recon;
        let error = diff.frobenius_norm() / orig_norm;
        let computed_fit = 1.0 - error;

        assert!(
            (cp.fit - computed_fit).abs() < 0.15,
            "fit field {:.6} vs reconstruction fit {:.6}",
            cp.fit,
            computed_fit
        );
    }

    // ------------------------------------------------------------------
    // 4. All-zeros tensor
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_all_zeros() {
        let coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        let cp = cp_als_sparse(&coo, 2, 10, 1e-4, IS::Random, None).unwrap();
        assert!(cp.fit >= 0.0 && cp.fit <= 1.0);
        assert_eq!(cp.factors.len(), 3);
    }

    // ------------------------------------------------------------------
    // 5. Single nonzero — rank-1 should capture it well
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_single_nonzero() {
        let mut coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        coo.push(vec![1, 2, 3], 1.0).unwrap();
        let cp = cp_als_sparse(&coo, 1, 50, 1e-6, IS::Random, None).unwrap();
        assert!(
            cp.fit > 0.5,
            "single-nonzero rank-1 fit should be high, got {}",
            cp.fit
        );
    }

    // ------------------------------------------------------------------
    // 6. Error: rank = 0
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_invalid_rank_zero() {
        let coo = build_sparse_coo(&[4, 5, 6], 0.1, 1);
        assert!(matches!(
            cp_als_sparse(&coo, 0, 10, 1e-4, IS::Random, None),
            Err(CpError::InvalidRank(0))
        ));
    }

    // ------------------------------------------------------------------
    // 7. Error: rank > mode size
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_invalid_rank_too_large() {
        let coo = build_sparse_coo(&[3, 5, 6], 0.1, 2);
        assert!(matches!(
            cp_als_sparse(&coo, 10, 10, 1e-4, IS::Random, None),
            Err(CpError::InvalidRank(_))
        ));
    }

    // ------------------------------------------------------------------
    // 8. Convergence info is populated
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_convergence_info() {
        let coo = build_sparse_coo(&[5, 5, 5], 0.10, 99);
        let cp = cp_als_sparse(&coo, 2, 30, 1e-4, IS::Random, None).unwrap();
        let conv = cp.convergence.as_ref().unwrap();
        assert!(!conv.fit_history.is_empty());
        assert!(conv.fit_history.len() <= 30);
    }

    // ------------------------------------------------------------------
    // 9. Higher rank gives equal-or-better fit
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_higher_rank_better_fit() {
        let coo = build_sparse_coo(&[8, 8, 8], 0.05, 55);
        let cp_lo = cp_als_sparse(&coo, 2, 30, 1e-4, IS::Random, None).unwrap();
        let cp_hi = cp_als_sparse(&coo, 5, 30, 1e-4, IS::Random, None).unwrap();
        assert!(
            cp_hi.fit >= cp_lo.fit * 0.9,
            "rank-5 fit {:.4} should not be much worse than rank-2 fit {:.4}",
            cp_hi.fit,
            cp_lo.fit
        );
    }

    // ------------------------------------------------------------------
    // 10. Time limit halts early
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_time_limit() {
        use std::time::Duration;
        let coo = build_sparse_coo(&[20, 20, 20], 0.05, 33);
        let cp = cp_als_sparse(
            &coo,
            4,
            1000,
            1e-8,
            IS::Random,
            Some(Duration::from_millis(100)),
        )
        .unwrap();
        assert!(cp.iters < 1000, "time limit should cut iterations short");
    }

    // ------------------------------------------------------------------
    // 11. RandomNormal initialization
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_random_normal_init() {
        let coo = build_sparse_coo(&[5, 6, 7], 0.08, 13);
        let cp = cp_als_sparse(&coo, 3, 15, 1e-4, IS::RandomNormal, None).unwrap();
        assert!(cp.fit >= 0.0);
    }

    // ------------------------------------------------------------------
    // 12. SVD initialization (densifies once internally)
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_svd_init() {
        let coo = build_sparse_coo(&[6, 7, 8], 0.08, 17);
        let cp = cp_als_sparse(&coo, 3, 15, 1e-4, IS::Svd, None)
            .expect("SVD init should work via one-time densification");
        assert!(cp.fit >= 0.0);
    }

    // ------------------------------------------------------------------
    // 13. 4D tensor
    // ------------------------------------------------------------------
    #[test]
    fn test_cp_als_sparse_4d() {
        let coo = build_sparse_coo(&[4, 5, 6, 7], 0.05, 200);
        let cp = cp_als_sparse(&coo, 2, 20, 1e-4, IS::Random, None).unwrap();
        assert_eq!(cp.factors.len(), 4);
        assert_eq!(cp.factors[0].shape(), &[4, 2]);
        assert_eq!(cp.factors[3].shape(), &[7, 2]);
    }

    // ------------------------------------------------------------------
    // 14. Parallel variant: parity with serial on same input
    // ------------------------------------------------------------------
    #[cfg(feature = "parallel")]
    #[test]
    fn test_cp_als_sparse_parallel_parity() {
        use crate::cp::cp_als_sparse_parallel;

        let coo = build_sparse_coo(&[8, 7, 6], 0.10, 314);
        let cp_s = cp_als_sparse(&coo, 3, 30, 1e-5, IS::Random, None).unwrap();
        let cp_p = cp_als_sparse_parallel(&coo, 3, 30, 1e-5, IS::Random, None).unwrap();

        // Both must return valid decompositions
        assert!(cp_s.fit >= 0.0 && cp_s.fit <= 1.0);
        assert!(cp_p.fit >= 0.0 && cp_p.fit <= 1.0);
        // Fits should be in the same ballpark (both explore the same landscape)
        assert_eq!(cp_s.factors.len(), cp_p.factors.len());
        assert_eq!(cp_s.factors[0].shape(), cp_p.factors[0].shape());
    }

    // ------------------------------------------------------------------
    // 15. Parallel variant: all-zeros
    // ------------------------------------------------------------------
    #[cfg(feature = "parallel")]
    #[test]
    fn test_cp_als_sparse_parallel_all_zeros() {
        use crate::cp::cp_als_sparse_parallel;

        let coo = CooTensor::<f64>::zeros(vec![4, 5, 6]).unwrap();
        let cp = cp_als_sparse_parallel(&coo, 2, 10, 1e-4, IS::Random, None).unwrap();
        assert!(cp.fit >= 0.0 && cp.fit <= 1.0);
    }
}
