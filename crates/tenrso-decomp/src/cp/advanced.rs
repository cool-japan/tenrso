//! Advanced CP decomposition algorithms
//!
//! Contains: cp_als_accelerated, cp_completion, cp_randomized, cp_als_incremental

use super::dimtree::AlsDimTree;
use super::els::{ErrorPolynomial, ELS_ALPHA_MAX, ELS_ALPHA_MIN, ELS_REFINE_ITERS};
use super::helpers::*;
use super::types::*;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use scirs2_core::random::thread_rng;
use scirs2_linalg::lstsq;
use std::iter::Sum;
use tenrso_core::DenseND;
use tenrso_kernels::mttkrp;

/// CP-ALS accelerated by an **exact Enhanced Line Search** (ELS) on the sweep's
/// extrapolation direction.
///
/// # Algorithm
///
/// Each iteration runs one exact Gauss-Seidel ALS sweep (the same dimension-tree engine
/// [`cp_als`](crate::cp_als) uses, `2·nnz·R`), takes the resulting *joint* direction
/// `D_k = A_k^new − A_k^prev`, and then **searches the line**
///
/// ```text
/// A_k(α) = A_k^prev + α · D_k        (all N factors move together)
/// ```
///
/// for the `α` that minimises `‖X − X̂(α)‖²`, and **applies that α**. `α = 1` reproduces
/// the plain sweep, `α > 1` extrapolates (this is what breaks CP-ALS *swamps*), and
/// `α < 1` damps when the joint move overshoots — which it can, because only mode `N−1`
/// is optimal for the post-sweep factor set.
///
/// The search is **exact, not a grid probe**: along that line the squared error is a
/// polynomial in `α` of degree `2N` whose `2N+1` coefficients are assembled once per
/// sweep from `N−1` extra MTTKRPs plus `R×R` Gram algebra (see the internal `els`
/// module). Evaluating the error at *any* `α` is then a Horner evaluation, so the global
/// minimiser on `[0, 4]` is found by rooting the derivative. Because `α = 1` is always in
/// the candidate set, the accelerated sweep is **provably never worse than a plain ALS
/// sweep**, and the final fit costs nothing extra — `p(α*)` *is* the squared error.
///
/// # What it actually buys you — measured, not asserted
///
/// One accelerated sweep costs `(N+1)·nnz·R` (the `2·nnz·R` sweep plus `(N−1)·nnz·R` of
/// exact-search MTTKRP probes) against plain CP-ALS's `2·nnz·R` — i.e. **2× per sweep for
/// a 3-way tensor, 2.5× for a 4-way one**. That per-sweep tax is the exact price of an
/// exact line search, and there is no cheaper exact variant: the mode-0 MTTKRP is a
/// degree-`N−1` matrix polynomial, so `N−1` probes are the interpolation floor.
///
/// The honest scorecard therefore has three separate axes. Measured on 40³ tensors,
/// **time to a common fit target**, median of 5, contended box
/// (`cargo run --release -p tenrso-decomp --example cp_els_bench`):
///
/// | tensor                       | sweeps→target | wall→target | final fit `cp_als` → accel |
/// |------------------------------|:-------------:|:-----------:|:--------------------------:|
/// | uniform random, rank 10      |   **1.8×**    |   0.9×      | 0.5108 → 0.5108            |
/// | clean low-rank + small noise |    1.0×       |   1.0×      | 0.2552 → 0.2552            |
/// | collinear swamp, `c = 0.90`  |   **2.4×**    |   0.9×      | 0.99995 → **1.00000**     |
/// | collinear swamp, `c = 0.99`  |   **2.4×**    | **1.3×**    | 0.99853 → **0.99895**     |
/// | swamp `c = 0.99` + noise     |   **1.7×**    |   1.0×      | 0.9486 → 0.9487           |
/// | collinear swamp, 4-way       |   **2.4×**    |   0.6×      | 0.99919 → **0.99999**     |
///
/// Read that as:
///
/// * **Iterations to a target: a real, consistent 1.7–2.4× fewer sweeps in swamps.** The
///   extrapolation direction is exactly what breaks the swamp.
/// * **Attainable quality: strictly higher in swamps.** In the same budget ELS reaches
///   fits plain ALS does not (`1.00000` vs `0.99995`, `0.99999` vs `0.99919`) — the
///   headline benefit, because in a deep swamp plain ALS can need *thousands* more sweeps
///   to close that gap.
/// * **Wall clock is roughly break-even, NOT a blanket win.** The 2×–2.5× per-sweep tax
///   eats most of the sweep-count saving: best case here is **1.3×** faster to target
///   (deep 3-way swamp), and it is a net **loss** (0.6×) on the 4-way tensor, where the
///   per-sweep penalty outgrows the sweep saving. There is **no** general wall-clock
///   speedup to promise, and this doc does not promise one.
///
/// **When to reach for it:** ill-conditioned / collinear **3-way** problems where plain
/// ALS stalls in a swamp and you care about the fit it cannot otherwise reach. On an easy,
/// well-separated low-rank tensor plain [`cp_als`](crate::cp_als) already converges in a
/// handful of sweeps and there is nothing to accelerate — ELS then merely pays its probe
/// overhead for no gain (and for order `N ≥ 4` that overhead makes it slower in wall
/// clock). What it *never* does is return a worse fit than [`cp_als`](crate::cp_als): the
/// search always includes `α = 1`, so every sweep is at least as good as the plain sweep
/// it replaces.
///
/// # Arguments
///
/// * `tensor` - Input tensor to decompose
/// * `rank` - CP rank (number of components)
/// * `max_iters` - Maximum number of ALS sweeps
/// * `tol` - Convergence tolerance for relative fit change
/// * `init` - Initialization strategy for factor matrices
/// * `time_limit` - Optional wall-clock limit
///
/// # Errors
///
/// Invalid rank or tolerance, an order-`< 2` tensor, or a failing linear-algebra solve.
///
/// # Complexity
///
/// Time: `O(max_iters · ((N+1)·nnz·R + N·Imax·R² + N·R³))`
/// Space: `O(N·Imax·R)` for the two factor sets plus `O(√nnz·R)` of tree working set.
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
/// println!("Converged in {} sweeps", cp.iters);
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
        + std::fmt::Display
        + 'static,
{
    cp_als_accelerated_traced(tensor, rank, max_iters, tol, init, time_limit)
        .map(|(decomp, _alphas)| decomp)
}

/// [`cp_als_accelerated`], additionally returning the **line-searched `α` of every
/// sweep**.
///
/// The public [`CpDecomp`] has nowhere to carry the step history, but the regression
/// tests have to be able to prove two things that no fit number can show on its own:
/// that the applied step *is* the searched optimum, and that the search really does
/// return `α < 1` when the full ALS step overshoots. Hence this crate-internal variant.
pub(crate) fn cp_als_accelerated_traced<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    time_limit: Option<std::time::Duration>,
) -> Result<(CpDecomp<T>, Vec<f64>), CpError>
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
        + std::fmt::Display
        + 'static,
{
    let start_time = std::time::Instant::now();

    if rank == 0 {
        return Err(CpError::InvalidRank(rank));
    }
    if tol <= 0.0 || tol >= 1.0 {
        return Err(CpError::InvalidTolerance(tol));
    }

    let tol_t: T = cast_f64(tol, "tol")?;

    let mut factors = initialize_factors(tensor, rank, init)?;
    let n_modes = factors.len();

    // Shape-only, built once, reused by every sweep.
    let tree = AlsDimTree::new(tensor.shape())?;
    let tensor_view = tensor.view();

    let tensor_norm_sq_t = compute_norm_squared(tensor);
    let tensor_norm_sq = tensor_norm_sq_t.to_f64().ok_or_else(|| {
        CpError::ShapeMismatch("tensor norm is not representable as f64".to_string())
    })?;
    let tensor_norm = tensor_norm_sq.sqrt();

    let mut fit = T::zero();
    let mut prev_fit = T::zero();
    let mut fit_history = Vec::with_capacity(max_iters);
    let mut alpha_history = Vec::with_capacity(max_iters);
    let mut oscillation_count = 0usize;
    let mut convergence_reason = ConvergenceReason::MaxIterations;
    let mut final_fit_change = T::zero();
    let mut iters = 0usize;

    for iter in 0..max_iters {
        iters = iter + 1;

        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                convergence_reason = ConvergenceReason::TimeLimit;
                break;
            }
        }

        let prev_factors: Vec<Array2<T>> = factors.clone();

        // ── One exact Gauss-Seidel ALS sweep (2·nnz·R on the dimension tree) ──────
        //
        // Mode 0 is visited first, so its MTTKRP is taken against the *pre-sweep*
        // factors 1..N-1 — which is exactly the α = 0 interpolation node the error
        // polynomial needs. Capturing it here makes that node free.
        let mut mttkrp_0: Option<Array2<T>> = None;
        {
            let mut update = |mode: usize,
                              mttkrp_result: &Array2<T>,
                              factors: &mut Vec<Array2<T>>|
             -> Result<(), CpError> {
                if mode == 0 {
                    mttkrp_0 = Some(mttkrp_result.clone());
                }
                let gram = compute_gram_hadamard(factors, mode);
                factors[mode] = solve_least_squares(mttkrp_result, &gram)?;
                Ok(())
            };
            tree.gauss_seidel_sweep(&tensor_view, &mut factors, &mut update)?;
        }
        let mttkrp_0 = mttkrp_0.ok_or_else(|| {
            CpError::ShapeMismatch("dimension-tree sweep never visited mode 0".to_string())
        })?;

        // ── The line search: exact, over the joint extrapolation direction ────────
        let polynomial = ErrorPolynomial::build(
            &tensor_view,
            &tree,
            &prev_factors,
            &factors,
            &mttkrp_0,
            tensor_norm_sq,
        )?;
        let (alpha, error_sq) = polynomial.minimize(ELS_ALPHA_MIN, ELS_ALPHA_MAX, ELS_REFINE_ITERS);
        alpha_history.push(alpha);

        // ── APPLY the searched optimum. The applied point IS the argmin. ──────────
        //
        // α = 1 is the plain sweep result, already in `factors`; anything else is a
        // genuine re-blend of the two factor sets. α < 1 damps, α > 1 extrapolates.
        if alpha != 1.0 {
            let alpha_t: T = cast_f64(alpha, "els alpha")?;
            for mode in 0..n_modes {
                let rows = factors[mode].shape()[0];
                for i in 0..rows {
                    for r in 0..rank {
                        let base = prev_factors[mode][[i, r]];
                        let step = factors[mode][[i, r]] - base;
                        factors[mode][[i, r]] = base + alpha_t * step;
                    }
                }
            }
        }

        // ── The fit is free: `error_sq` *is* ‖X − X̂(α*)‖². No reconstruction. ─────
        let fit_f64 = if tensor_norm > 0.0 {
            1.0 - error_sq.max(0.0).sqrt() / tensor_norm
        } else {
            0.0
        };
        let fit_f64 = if fit_f64.is_finite() {
            fit_f64.clamp(0.0, 1.0)
        } else {
            0.0
        };
        fit = cast_f64(fit_f64, "fit")?;
        fit_history.push(fit);

        if iter > 0 {
            final_fit_change = (fit - prev_fit).abs();

            // With an exact line search the error cannot increase, so this should stay
            // at zero; it is kept as an honest tripwire, not as a control knob.
            if fit < prev_fit {
                oscillation_count += 1;
            }

            let relative_change = final_fit_change / (prev_fit.abs() + cast_lit::<T, _>(1e-10_f64));
            if relative_change < tol_t {
                convergence_reason = ConvergenceReason::FitTolerance;
                break;
            }

            if oscillation_count > 5 && iter > 10 {
                convergence_reason = ConvergenceReason::Oscillation;
                break;
            }
        }

        prev_fit = fit;
    }

    let decomp = CpDecomp {
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
    };

    Ok((decomp, alpha_history))
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

    let tol_t: T = cast_f64(tol, "tol")?;

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

            // ── Per-row weighted normal-equation solve ────────────────────────────
            //
            // Weighted CP completion is *not* one least-squares problem per mode: the
            // observation mask varies from row to row, so each row `i` of the factor
            // has its own R×R normal system built only from the columns observed in
            // that very row:
            //
            //     G_i = Σ_{j : mask_unfolded[i,j] > 0}  kr[j,:]ᵀ kr[j,:]      (R×R)
            //     b_i = masked MTTKRP row i  (already assembled above)         (R)
            //     factors[mode][i,:] = G_i^{-1} · b_i
            //
            // Aggregating one Gram over *all* observed (i,j) pairs and applying its
            // single inverse to every row solves a different, wrong problem — and for
            // an all-ones mask it inflates the Gram by a factor of `mode_size`, so even
            // a fully observed tensor is not reproduced. The per-row form is the correct
            // cost of weighted completion: O(mode_size · (R²·nnz_row + R³)).
            //
            // A Tikhonov ridge `λ_i · I` stabilises rows with few (or zero) observed
            // fibres, where `G_i` is rank-deficient. It is scaled to the row's own Gram,
            // `λ_i = 1e-8 · tr(G_i)/R`, plus a fixed `1e-12` floor so that a completely
            // unobserved row (`G_i = 0`, `b_i = 0`) still yields the finite minimum-norm
            // solution `0` instead of a singular solve. On a well-observed row this ridge
            // is a ~1e-8 relative perturbation, negligible against the solution.
            let mut factor_new = Array2::<T>::zeros((mode_size, rank));
            let rank_t: T = cast_lit(rank);
            let ridge_rel: T = cast_lit(1e-8_f64);
            let ridge_floor: T = cast_lit(1e-12_f64);

            for i in 0..mode_size {
                let mut gram_row = Array2::<T>::zeros((rank, rank));
                for j in 0..kr.nrows() {
                    if mask_unfolded[[i, j]] > T::zero() {
                        for r1 in 0..rank {
                            let kr_j_r1 = kr[[j, r1]];
                            for r2 in 0..rank {
                                gram_row[[r1, r2]] += kr_j_r1 * kr[[j, r2]];
                            }
                        }
                    }
                }

                let mut trace = T::zero();
                for r in 0..rank {
                    trace += gram_row[[r, r]];
                }
                let ridge = ridge_rel * trace / rank_t + ridge_floor;
                for r in 0..rank {
                    gram_row[[r, r]] += ridge;
                }

                let b_i = mttkrp_result.row(i).to_owned();
                let solution =
                    lstsq(&gram_row.view(), &b_i.view(), None).map_err(CpError::LinalgError)?;
                for r in 0..rank {
                    factor_new[[i, r]] = solution.x[r];
                }
            }

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
            let relative_change = fit_change / (prev_fit.abs() + cast_lit::<T, _>(1e-10_f64));

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
    use scirs2_core::random::Distribution;

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
    let normal = make_normal(0.0, 1.0)?;
    let tol_t: T = cast_f64(tol, "tol")?;

    // ALS iterations with randomized sketching
    for iter in 0..max_iters {
        iters = iter + 1;

        for mode in 0..n_modes {
            let kr = compute_khatri_rao_except(&factors, mode);
            let kr_rows = kr.shape()[0];

            let mut omega = Array2::<T>::zeros((kr_rows, sketch_size));
            for i in 0..kr_rows {
                for j in 0..sketch_size {
                    omega[[i, j]] = cast_lit(normal.sample(&mut rng));
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
            if iter > 0 && fit_change < tol_t {
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
            let old_rows_t: T = cast_lit(old_rows);
            for i in old_rows..total_rows {
                for j in 0..rank {
                    let mut col_mean = T::zero();
                    for k in 0..old_rows {
                        col_mean += current.factors[update_mode][[k, j]];
                    }
                    col_mean /= old_rows_t;

                    let noise: T = cast_lit(rng.random::<f64>() * 0.1 - 0.05);
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
    let tol_t: T = cast_f64(tol, "tol")?;

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
            if fit_change < tol_t {
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

#[cfg(test)]
mod els_regression {
    //! Regression tests for the ELS line search in [`cp_als_accelerated`].
    //!
    //! # What broke, and what these tests pin down
    //!
    //! The previous implementation computed a step size and then **threw it away**: it
    //! applied `A_prev + (1 + α_ls·α)·D` instead of the searched point `A_prev + α_ls·D`,
    //! re-purposing the line-searched value as a *momentum gain on top of an already-full
    //! ALS step*. With `α ∈ [0.1, 0.9]` (and `α_ls` provably identically `1`, since a
    //! single-mode search along the ALS direction can only ever return `1`), the effective
    //! step was confined to `[1.1, 1.9]`: **strictly greater than 1, always**. A line
    //! search that cannot return a step below 1 cannot damp, and damping is the one thing
    //! a line search exists to do.
    //!
    //! `OLD_STEP_RANGE` below is that reachable interval. The tests show the exact error
    //! polynomial takes its minimum *outside* it — so the old code was structurally
    //! forced onto a strictly worse point.

    use super::super::els::{ErrorPolynomial, ELS_ALPHA_MAX, ELS_ALPHA_MIN, ELS_REFINE_ITERS};
    use super::*;
    use crate::cp_als;
    use scirs2_core::ndarray_ext::{Array, IxDyn};

    /// Every step the *old* implementation could possibly take: `1 + α_ls·α` with
    /// `α_ls ≡ 1` and `α` clamped to `[0.1, 0.9]`.
    const OLD_STEP_RANGE: (f64, f64) = (1.1, 1.9);

    /// Deterministic data generator. The regression must be reproducible, and
    /// `scirs2_core::random`'s thread RNG is not seedable from a test.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// A CP tensor whose factor columns have pairwise correlation `collinearity`.
    ///
    /// High collinearity is the classical CP-ALS **swamp**: the Gram matrices go
    /// near-singular, the per-mode solves take long swings, and a sweep's modes end up
    /// strongly coupled — so the *joint* move of all `N` factors at once genuinely
    /// overshoots and has to be damped. This is the regime the accelerated driver exists
    /// for, and the regime in which the old always-overstep rule did the most damage.
    fn collinear_cp_tensor(
        shape: &[usize],
        rank: usize,
        collinearity: f64,
        seed: u64,
    ) -> DenseND<f64> {
        let mut rng = Lcg::new(seed);
        let n_modes = shape.len();

        let mut factors: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_modes);
        for &dim in shape {
            let base: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
            let mut columns = Vec::with_capacity(rank);
            for _ in 0..rank {
                let independent: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
                let mut column: Vec<f64> = (0..dim)
                    .map(|i| {
                        collinearity * base[i]
                            + (1.0 - collinearity * collinearity).sqrt() * independent[i]
                    })
                    .collect();
                let norm = column.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                for value in &mut column {
                    *value /= norm;
                }
                columns.push(column);
            }
            factors.push(columns);
        }

        let mut data = Array::<f64, IxDyn>::zeros(IxDyn(shape));
        let numel: usize = shape.iter().product();
        let mut index = vec![0usize; n_modes];
        for flat in 0..numel {
            let mut remainder = flat;
            for mode in (0..n_modes).rev() {
                index[mode] = remainder % shape[mode];
                remainder /= shape[mode];
            }
            let mut value = 0.0;
            for r in 0..rank {
                let mut term = 1.0;
                for (mode, &i) in index.iter().enumerate() {
                    term *= factors[mode][r][i];
                }
                value += term;
            }
            data[IxDyn(&index)] = value;
        }

        DenseND::from_array(data)
    }

    /// The regression fixture: a 3-way swamp. `Nnsvd` gives a deterministic,
    /// *non-orthogonal* (hence strongly coupled) start, which is what makes the joint ALS
    /// direction overshoot.
    fn overshoot_fixture() -> (DenseND<f64>, usize, InitStrategy) {
        (
            collinear_cp_tensor(&[8, 8, 8], 3, 0.9, 1),
            3,
            InitStrategy::Nnsvd,
        )
    }

    /// Re-run the driver's sweeps by hand, yielding, for every sweep, the exact error
    /// polynomial and the factor set the driver ended that sweep on.
    fn replay(
        tensor: &DenseND<f64>,
        rank: usize,
        init: InitStrategy,
        sweeps: usize,
    ) -> Vec<(ErrorPolynomial, f64)> {
        let tree = AlsDimTree::new(tensor.shape()).expect("tree");
        let tensor_view = tensor.view();
        let norm_sq = compute_norm_squared(tensor);
        let mut factors = initialize_factors(tensor, rank, init).expect("init");
        let n_modes = factors.len();

        let mut out = Vec::with_capacity(sweeps);
        for _ in 0..sweeps {
            let prev: Vec<Array2<f64>> = factors.clone();

            let mut mttkrp_0: Option<Array2<f64>> = None;
            {
                let mut update = |mode: usize,
                                  m: &Array2<f64>,
                                  f: &mut Vec<Array2<f64>>|
                 -> Result<(), CpError> {
                    if mode == 0 {
                        mttkrp_0 = Some(m.clone());
                    }
                    let gram = compute_gram_hadamard(f, mode);
                    f[mode] = solve_least_squares(m, &gram)?;
                    Ok(())
                };
                tree.gauss_seidel_sweep(&tensor_view, &mut factors, &mut update)
                    .expect("sweep");
            }
            let mttkrp_0 = mttkrp_0.expect("mode 0 is visited first");

            let polynomial =
                ErrorPolynomial::build(&tensor_view, &tree, &prev, &factors, &mttkrp_0, norm_sq)
                    .expect("polynomial");
            let (alpha, _) = polynomial.minimize(ELS_ALPHA_MIN, ELS_ALPHA_MAX, ELS_REFINE_ITERS);

            for mode in 0..n_modes {
                let rows = factors[mode].shape()[0];
                factors[mode] = Array2::<f64>::from_shape_fn((rows, rank), |(i, r)| {
                    prev[mode][[i, r]] + alpha * (factors[mode][[i, r]] - prev[mode][[i, r]])
                });
            }

            out.push((polynomial, alpha));
        }
        out
    }

    /// **The bug.** The step the driver applies must *be* the step the search found — not
    /// that step used as a momentum gain on top of a full ALS step.
    ///
    /// Checked two ways: the applied `α` is the global minimiser of the exact error over
    /// the whole search interval, and the driver's own `α` history agrees with an
    /// independent replay of the same sweeps.
    #[test]
    fn accelerated_applies_exactly_the_searched_optimum() {
        let (tensor, rank, init) = overshoot_fixture();
        let sweeps = 40;

        let (_decomp, driver_alphas) =
            cp_als_accelerated_traced(&tensor, rank, sweeps, 1e-12, init, None)
                .expect("accelerated CP should succeed");
        let replayed = replay(&tensor, rank, init, driver_alphas.len());

        for (sweep, ((polynomial, alpha), &driver_alpha)) in
            replayed.iter().zip(driver_alphas.iter()).enumerate()
        {
            assert!(
                (alpha - driver_alpha).abs() < 1e-12,
                "sweep {sweep}: driver applied alpha {driver_alpha}, search says {alpha}"
            );

            // The applied point is the argmin of the exact error over [0, 4].
            let applied = polynomial.eval(*alpha);
            for step in 0..=400 {
                let candidate =
                    ELS_ALPHA_MIN + (ELS_ALPHA_MAX - ELS_ALPHA_MIN) * (step as f64) / 400.0;
                let value = polynomial.eval(candidate);
                assert!(
                    applied <= value + 1e-9 * applied.abs().max(1.0),
                    "sweep {sweep}: applied alpha {alpha} gives error {applied:.12e}, but \
                     alpha {candidate} gives a smaller error {value:.12e}"
                );
            }
        }
    }

    /// **The concrete failure the old code could not avoid.** On an overshooting sweep the
    /// exact optimum is *below 1* — the full ALS step is too long and must be damped. The
    /// old rule was confined to `[1.1, 1.9]`, so it could not merely fail to damp: every
    /// step available to it was strictly worse than the one this search finds.
    #[test]
    fn overshooting_sweep_is_damped_below_one() {
        let (tensor, rank, init) = overshoot_fixture();
        let replayed = replay(&tensor, rank, init, 40);

        let damped: Vec<usize> = replayed
            .iter()
            .enumerate()
            .filter(|(_, (_, alpha))| *alpha < 1.0)
            .map(|(sweep, _)| sweep)
            .collect();

        assert!(
            !damped.is_empty(),
            "the fixture must actually overshoot somewhere: alphas = {:?}",
            replayed.iter().map(|(_, a)| *a).collect::<Vec<_>>()
        );

        for &sweep in &damped {
            let (polynomial, alpha) = &replayed[sweep];
            let damped_error = polynomial.eval(*alpha);

            // Damping beats the plain, undamped ALS sweep.
            let full_step_error = polynomial.eval(1.0);
            assert!(
                damped_error < full_step_error,
                "sweep {sweep}: alpha {alpha} < 1 must strictly beat the full ALS step"
            );

            // ...and beats *everything* the old always-overstep rule could have chosen.
            let (lo, hi) = OLD_STEP_RANGE;
            for step in 0..=100 {
                let old = lo + (hi - lo) * (step as f64) / 100.0;
                assert!(
                    damped_error < polynomial.eval(old),
                    "sweep {sweep}: the old rule's step {old} is not worse than the \
                     searched optimum {alpha} — the fixture does not demonstrate overshoot"
                );
            }
        }
    }

    /// End-to-end, through the public API: an exact line search that always includes
    /// `α = 1` cannot do worse than the plain sweep it accelerates, so the accelerated
    /// driver must never return a worse fit than [`cp_als`].
    ///
    /// The old code failed this: forced to overstep every sweep, it landed at fit 0.9936
    /// where plain CP-ALS reached 0.9995 on the 40³ `c = 0.9` swamp.
    #[test]
    fn accelerated_fit_is_never_worse_than_plain_cp_als() {
        for (collinearity, seed) in [(0.9f64, 1u64), (0.9, 3), (0.95, 5), (0.99, 7)] {
            let tensor = collinear_cp_tensor(&[12, 12, 12], 3, collinearity, seed);

            for init in [InitStrategy::Svd, InitStrategy::Nnsvd] {
                let plain = cp_als(&tensor, 3, 60, 1e-10, init, None).expect("cp_als");
                let accelerated =
                    cp_als_accelerated(&tensor, 3, 60, 1e-10, init, None).expect("accelerated");

                assert!(
                    accelerated.fit >= plain.fit - 1e-9,
                    "c={collinearity} seed={seed} init={init:?}: accelerated fit {:.9} is \
                     worse than plain cp_als {:.9}",
                    accelerated.fit,
                    plain.fit
                );
            }
        }
    }

    /// The fit the driver reports is derived from the ELS polynomial rather than from a
    /// reconstruction, so it has to be checked against an actual reconstruction.
    #[test]
    fn reported_fit_matches_an_explicit_reconstruction() {
        let (tensor, rank, init) = overshoot_fixture();
        let decomp = cp_als_accelerated(&tensor, rank, 25, 1e-12, init, None).expect("accelerated");

        let reconstructed = compute_reconstruction(&decomp.factors).expect("reconstruct");
        let mut error_sq = 0.0f64;
        for (&x, &y) in tensor.view().iter().zip(reconstructed.view().iter()) {
            error_sq += (x - y) * (x - y);
        }
        let expected = 1.0 - error_sq.sqrt() / tensor.frobenius_norm();

        assert!(
            (decomp.fit - expected).abs() < 1e-9,
            "reported fit {:.12} != reconstruction fit {:.12}",
            decomp.fit,
            expected
        );
    }

    /// An exact line search cannot increase the error, so the fit history must be
    /// monotonically non-decreasing — a property the old momentum rule did not have.
    #[test]
    fn fit_history_is_monotone() {
        let (tensor, rank, init) = overshoot_fixture();
        let decomp = cp_als_accelerated(&tensor, rank, 40, 1e-12, init, None).expect("accelerated");
        let convergence = decomp.convergence.expect("convergence info");

        for window in convergence.fit_history.windows(2) {
            assert!(
                window[1] >= window[0] - 1e-9,
                "fit went backwards: {:?}",
                convergence.fit_history
            );
        }
        assert_eq!(
            convergence.oscillation_count, 0,
            "an exact line search must not oscillate"
        );
    }
}

#[cfg(test)]
mod completion_regression {
    //! Regression tests for [`cp_completion`].
    //!
    //! # What broke, and what these tests pin down
    //!
    //! Weighted CP completion fits a CP model to a *partially observed* tensor. For
    //! each mode, row `i` of the factor is the solution of its **own** R×R normal
    //! system, assembled only from the fibres observed *in that row*:
    //!
    //! ```text
    //! G_i = Σ_{j : mask[i,j] > 0}  kr[j,:]ᵀ kr[j,:]        b_i = masked-MTTKRP row i
    //! factors[mode][i,:] = G_i⁻¹ · b_i
    //! ```
    //!
    //! The previous implementation instead built a **single** R×R Gram aggregated over
    //! *all* observed `(i,j)` pairs and applied its one inverse to every row. Because the
    //! mask varies per row, that solves the wrong least-squares problem for every row but
    //! the (accidental) uniform-mask one. The tell-tale side effect: on an *all-ones*
    //! mask the aggregated Gram equals `mode_size · (krᵀkr)`, so the scale is wrong and a
    //! *fully observed* tensor is not even reproduced — completion on an all-ones mask
    //! landed at reconstruction relative error ≈ 0.83 where plain [`cp_als`] reaches
    //! machine epsilon.
    //!
    //! Each test below **fails against the aggregated-Gram code** and passes against the
    //! per-row solve.

    use super::*;
    use crate::cp_als;
    use scirs2_core::ndarray_ext::{Array, IxDyn};

    /// Deterministic LCG — `scirs2_core::random`'s thread RNG is not seedable from a
    /// test, and these regressions must be reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// An **exact** rank-`rank` tensor built from random Gaussian factors. A correct
    /// completion of an exact low-rank tensor with enough observations must recover it
    /// to solver precision.
    fn exact_cp_tensor(shape: &[usize], rank: usize, seed: u64) -> DenseND<f64> {
        let mut rng = Lcg::new(seed);
        let factors: Vec<Array2<f64>> = shape
            .iter()
            .map(|&dim| Array2::from_shape_fn((dim, rank), |_| rng.normal()))
            .collect();
        compute_reconstruction(&factors).expect("reconstruct exact tensor")
    }

    /// `truth` with every entry where `mask == 0` zeroed — the honest completion input:
    /// the solver never sees a held-out value.
    fn apply_mask(truth: &DenseND<f64>, mask: &DenseND<f64>) -> DenseND<f64> {
        let mut arr = truth.view().to_owned();
        for (a, &m) in arr.iter_mut().zip(mask.view().iter()) {
            if m == 0.0 {
                *a = 0.0;
            }
        }
        DenseND::from_array(arr)
    }

    /// Relative Frobenius error over *all* entries.
    fn relative_error(truth: &DenseND<f64>, recon: &DenseND<f64>) -> f64 {
        let mut err = 0.0;
        let mut nrm = 0.0;
        for (&t, &r) in truth.view().iter().zip(recon.view().iter()) {
            err += (t - r) * (t - r);
            nrm += t * t;
        }
        (err / nrm).sqrt()
    }

    /// Relative Frobenius error restricted to the held-out (`mask == 0`) entries — the
    /// only quantity that measures whether completion actually *completed*.
    fn missing_relative_error(
        truth: &DenseND<f64>,
        recon: &DenseND<f64>,
        mask: &DenseND<f64>,
    ) -> f64 {
        let mut err = 0.0;
        let mut nrm = 0.0;
        for ((&t, &r), &m) in truth
            .view()
            .iter()
            .zip(recon.view().iter())
            .zip(mask.view().iter())
        {
            if m == 0.0 {
                err += (t - r) * (t - r);
                nrm += t * t;
            }
        }
        (err / nrm).sqrt()
    }

    /// **The key regression.** With an all-ones mask, per-row completion is *identical*
    /// to plain [`cp_als`] (every row's Gram is the same full `krᵀkr`, which equals the
    /// Hadamard-of-Grams `cp_als` uses). So it must reproduce the exact tensor to solver
    /// precision **and** match `cp_als`'s reconstruction. The aggregated-Gram code fails
    /// this catastrophically — it inflates the Gram by `mode_size` and lands near rel
    /// error 0.83.
    #[test]
    fn all_ones_mask_reproduces_cp_als() {
        let shape = [8, 7, 6];
        let rank = 2;
        let tensor = exact_cp_tensor(&shape, rank, 1);
        let mask = DenseND::from_array(Array::<f64, IxDyn>::ones(IxDyn(&shape)));

        let completion = cp_completion(&tensor, &mask, rank, 300, 1e-10, InitStrategy::Svd)
            .expect("cp_completion");
        let als = cp_als(&tensor, rank, 300, 1e-10, InitStrategy::Svd, None).expect("cp_als");

        let recon_c = compute_reconstruction(&completion.factors).expect("recon completion");
        let recon_a = compute_reconstruction(&als.factors).expect("recon cp_als");

        let err_truth = relative_error(&tensor, &recon_c);
        assert!(
            err_truth < 1e-6,
            "all-ones completion rel error {err_truth:.3e} is not < 1e-6 \
             (aggregated-Gram bug lands near 0.83)"
        );

        let err_vs_als = relative_error(&recon_a, &recon_c);
        assert!(
            err_vs_als < 1e-6,
            "completion vs cp_als reconstruction differ by {err_vs_als:.3e} (not < 1e-6)"
        );
    }

    /// A high-observation (≈88%) random mask on an exact rank-2 tensor. Every fibre is
    /// then observed far more than `rank` times, so each per-row Gram is well determined
    /// and the completion fixed point is the true tensor. Held-out entries must be
    /// recovered to small error. The aggregated-Gram code cannot (oracle: ≈0.88 on the
    /// missing entries). Threshold `1e-4` sits ~4 orders below the broken code and well
    /// above solver precision for an exact low-rank tensor at this observation fraction.
    #[test]
    fn high_observation_recovers_missing_entries() {
        let shape = [10, 9, 8];
        let rank = 2;
        let truth = exact_cp_tensor(&shape, rank, 7);

        let observed_fraction = 0.88;
        let mut rng = Lcg::new(123);
        let mut mask_arr = Array::<f64, IxDyn>::zeros(IxDyn(&shape));
        {
            let mut mask_it = mask_arr.iter_mut();
            for _ in truth.view().iter() {
                let m = mask_it.next().expect("mask length matches tensor");
                if rng.uniform() < observed_fraction {
                    *m = 1.0;
                }
            }
        }
        let mask = DenseND::from_array(mask_arr);
        let observed = apply_mask(&truth, &mask);

        let completion = cp_completion(&observed, &mask, rank, 800, 1e-12, InitStrategy::Svd)
            .expect("cp_completion");
        let recon = compute_reconstruction(&completion.factors).expect("recon");

        let missing = missing_relative_error(&truth, &recon, &mask);
        assert!(
            missing < 1e-4,
            "held-out entries recovered only to rel error {missing:.3e} (not < 1e-4)"
        );
    }

    /// A low-observation (≈20%) mask with a mode-0 row forced to be *entirely* unobserved,
    /// so its per-row Gram is `G = 0` and only the ridge keeps the solve well posed. The
    /// solve must produce finite factors and a finite fit in `[0, 1]` — no NaN/inf from a
    /// singular system. Exercises the `1e-12` ridge floor directly.
    #[test]
    fn degenerate_rows_exercise_ridge_without_nan() {
        let shape = [6, 5, 4];
        let rank = 2;
        let truth = exact_cp_tensor(&shape, rank, 11);

        let mut rng = Lcg::new(99);
        let mut mask_arr = Array::<f64, IxDyn>::zeros(IxDyn(&shape));
        {
            let mut mask_it = mask_arr.iter_mut();
            for _ in truth.view().iter() {
                let m = mask_it.next().expect("mask length matches tensor");
                if rng.uniform() < 0.2 {
                    *m = 1.0;
                }
            }
        }
        // Force the entire mode-0 row 0 to be unobserved: G_0 becomes pure ridge.
        for (idx, m) in mask_arr.indexed_iter_mut() {
            if idx[0] == 0 {
                *m = 0.0;
            }
        }
        let mask = DenseND::from_array(mask_arr);
        let observed = apply_mask(&truth, &mask);

        let completion = cp_completion(&observed, &mask, rank, 50, 1e-8, InitStrategy::Random)
            .expect("cp_completion");

        for factor in &completion.factors {
            for &v in factor.iter() {
                assert!(
                    v.is_finite(),
                    "factor entry {v} is not finite (ridge failed)"
                );
            }
        }
        assert!(
            completion.fit.is_finite(),
            "fit {} is not finite",
            completion.fit
        );
        assert!(
            (0.0..=1.0).contains(&completion.fit),
            "fit {} is outside [0, 1]",
            completion.fit
        );
    }
}
