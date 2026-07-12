//! Core CP-ALS algorithms: standard, scheme-selectable, and constrained.
//!
//! # Sweep engine
//!
//! Both [`cp_als`] and [`cp_als_constrained`] run their sweeps on the
//! **Gauss-Seidel-preserving dimension tree** in [`super::dimtree`]. That is an
//! implementation change only: the sequence of iterates is the classical
//! Gauss-Seidel ALS sequence — mode `k` is still updated against the factors
//! `0..k` that this very sweep already refreshed — while the per-sweep cost drops
//! from `N·nnz·R` to `2·nnz·R` and `N` full tensor copies are eliminated. See the
//! [`super::dimtree`] module docs for why the tree's shared partials survive the
//! mid-sweep factor writes.
//!
//! The genuinely different algorithm — Jacobi / "fast ALS", where every mode of a
//! sweep sees the *frozen* previous factor set — is reachable only through the
//! explicit [`UpdateScheme::Jacobi`] opt-in on [`cp_als_with_scheme`].

use super::dimtree::AlsDimTree;
use super::helpers::*;
use super::types::*;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::numeric::{Float, FloatConst, NumAssign, NumCast};
use std::iter::Sum;
use tenrso_core::DenseND;

/// Shared per-iteration convergence bookkeeping for the ALS drivers.
///
/// Extracted verbatim from the original in-line logic so that every entry point
/// keeps byte-identical stopping behaviour (fit tolerance, oscillation detection,
/// history recording).
struct ConvergenceTracker<T> {
    fit_history: Vec<T>,
    prev_fit: T,
    fit: T,
    oscillation_count: usize,
    reason: ConvergenceReason,
    final_fit_change: T,
    iters: usize,
}

impl<T: Float> ConvergenceTracker<T> {
    fn new(max_iters: usize) -> Self {
        Self {
            fit_history: Vec::with_capacity(max_iters),
            prev_fit: T::zero(),
            fit: T::zero(),
            oscillation_count: 0,
            reason: ConvergenceReason::MaxIterations,
            final_fit_change: T::zero(),
            iters: 0,
        }
    }

    /// Record iteration `iter`'s fit; return `true` when the ALS loop must stop.
    fn record(&mut self, iter: usize, fit: T, tol: T) -> bool {
        self.fit = fit;
        self.fit_history.push(fit);

        if iter > 0 && fit < self.prev_fit {
            self.oscillation_count += 1;
        }

        let fit_change = (fit - self.prev_fit).abs();
        self.final_fit_change = fit_change;

        if iter > 0 && fit_change < tol {
            self.reason = ConvergenceReason::FitTolerance;
            return true;
        }

        if self.oscillation_count > 5 && iter > 10 {
            self.reason = ConvergenceReason::Oscillation;
            return true;
        }

        self.prev_fit = fit;
        false
    }

    fn finish(self, factors: Vec<Array2<T>>) -> CpDecomp<T> {
        CpDecomp {
            factors,
            weights: None,
            fit: self.fit,
            iters: self.iters,
            convergence: Some(ConvergenceInfo {
                fit_history: self.fit_history,
                reason: self.reason,
                oscillated: self.oscillation_count > 0,
                oscillation_count: self.oscillation_count,
                final_fit_change: self.final_fit_change,
            }),
        }
    }
}

/// Validate the rank/tolerance arguments shared by every CP-ALS entry point.
fn validate_cp_args<T>(tensor: &DenseND<T>, rank: usize, tol: f64) -> Result<T, CpError>
where
    T: Float + NumCast,
{
    if rank == 0 {
        return Err(CpError::InvalidRank(rank));
    }

    for &mode_size in tensor.shape().iter() {
        if rank > mode_size {
            return Err(CpError::InvalidRank(rank));
        }
    }

    if !(0.0..1.0).contains(&tol) {
        return Err(CpError::InvalidTolerance(tol));
    }

    cast_f64(tol, "tol")
}

/// Compute CP-ALS decomposition of a tensor
///
/// Classical **Gauss-Seidel** alternating least squares: within one sweep, mode `k`
/// is updated against the factors `0..k` that the same sweep has already refreshed
/// and the factors `k+1..N-1` left by the previous sweep. This is
/// [`UpdateScheme::GaussSeidel`], and it is what this function has always done and
/// always will do; [`cp_als_with_scheme`] exists for anything else.
///
/// Internally the sweep runs on a dimension tree that shares partial contractions
/// across all `N` modes *without* relaxing that sequential dependency, so the sweep
/// costs `2·nnz·R` rather than `N·nnz·R` (see the internal `dimtree` module). The iterates are
/// unchanged up to floating-point summation order.
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
/// - The tensor has order < 2
/// - Linear algebra operations fail
///
/// # Complexity
///
/// Time: O(max_iters * (2 * nnz * R + N * Imax * R^2 + N * R^3))
/// Space: O(sqrt(nnz) * R) working set on top of O(N * Imax * R) factor matrices
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
    cp_als_with_scheme(
        tensor,
        rank,
        max_iters,
        tol,
        init,
        UpdateScheme::GaussSeidel,
        time_limit,
    )
}

/// CP-ALS with an explicitly chosen [`UpdateScheme`].
///
/// **Read [`UpdateScheme`] before passing anything other than
/// [`UpdateScheme::GaussSeidel`]** — the two schemes are different algorithms, not
/// different implementations. `GaussSeidel` reproduces [`cp_als`] exactly.
///
/// # Why you would reach for `Jacobi`
///
/// Only for its structural property: the `N` mode updates of a Jacobi sweep are
/// mutually independent, so they can be farmed out concurrently (across threads,
/// nodes, or devices). Within a single process this crate's Gauss-Seidel path is
/// already `2·nnz·R` per sweep and gets the fit for free, so `Jacobi` is strictly
/// *more* expensive per sweep here (it needs one extra `nnz·R` MTTKRP to evaluate
/// the fit, because after a Jacobi sweep no intermediate is consistent with the new
/// factors) **and** it makes less progress per sweep, since every mode is updated
/// against stale partners. Expect it to need more sweeps to reach a given fit, and
/// to be more prone to the oscillation the convergence tracker watches for.
///
/// # Arguments
///
/// Same as [`cp_als`], plus `scheme`.
///
/// # Errors
///
/// As [`cp_als`].
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::cp::{cp_als_with_scheme, InitStrategy, UpdateScheme};
///
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
///
/// // Identical to `cp_als`.
/// let gs = cp_als_with_scheme(
///     &tensor, 5, 50, 1e-4, InitStrategy::Random, UpdateScheme::GaussSeidel, None,
/// ).unwrap();
///
/// // A DIFFERENT algorithm: every mode sees the frozen previous factor set.
/// let jacobi = cp_als_with_scheme(
///     &tensor, 5, 50, 1e-4, InitStrategy::Random, UpdateScheme::Jacobi, None,
/// ).unwrap();
///
/// assert!(gs.fit >= 0.0 && jacobi.fit >= 0.0);
/// ```
pub fn cp_als_with_scheme<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    scheme: UpdateScheme,
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
    let tol_t: T = validate_cp_args(tensor, rank, tol)?;
    let n_modes = tensor.rank();

    let mut factors = initialize_factors(tensor, rank, init)?;
    let tensor_norm_sq = compute_norm_squared(tensor);
    let tensor_view = tensor.view();

    // The tree depends only on the tensor SHAPE, so it is built once here and
    // reused by every sweep (O(N) to build, holds no tensor data).
    let tree = AlsDimTree::new(tensor.shape())?;

    let mut tracker = ConvergenceTracker::<T>::new(max_iters);
    let start_time = std::time::Instant::now();

    for iter in 0..max_iters {
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                tracker.reason = ConvergenceReason::TimeLimit;
                break;
            }
        }

        tracker.iters = iter + 1;

        let fit = match scheme {
            UpdateScheme::GaussSeidel => {
                // The tree hands each mode the MTTKRP that classical ALS would have
                // computed there, i.e. against the factors this sweep already wrote.
                let mut update = |mode: usize,
                                  mttkrp_result: &Array2<T>,
                                  factors: &mut Vec<Array2<T>>|
                 -> Result<(), CpError> {
                    let gram = compute_gram_hadamard(factors, mode);
                    factors[mode] = solve_least_squares(mttkrp_result, &gram)?;
                    Ok(())
                };
                let last_mttkrp =
                    tree.gauss_seidel_sweep(&tensor_view, &mut factors, &mut update)?;

                // `last_mttkrp` is mode N-1's, taken against the now-final factors
                // 0..N-2, so it supplies <X, [[A]]> exactly. No extra tensor pass.
                compute_fit_from_mttkrp(&factors, &last_mttkrp, n_modes - 1, tensor_norm_sq)
            }
            UpdateScheme::Jacobi => {
                let (new_factors, mttkrp_0) = jacobi_sweep(tensor, &tree, &tensor_view, &factors)?;
                factors = new_factors;

                // `mttkrp_0` does not involve factor 0, and `jacobi_sweep` only ever
                // rescales factor 0 after taking it — so it is still exactly the
                // mode-0 MTTKRP of the returned factors, and the fit is consistent.
                compute_fit_from_mttkrp(&factors, &mttkrp_0, 0, tensor_norm_sq)
            }
        };

        if tracker.record(iter, fit, tol_t) {
            break;
        }
    }

    Ok(tracker.finish(factors))
}

/// One Jacobi sweep's worth of MTTKRPs: all `N` modes against the frozen factor
/// set, taken in a single shared pass by the dense dimension-tree kernel
/// [`tenrso_kernels::DimTree::mttkrp_all`].
///
/// This is the kernel API in its natural form — every leaf against one snapshot of
/// the factors — which is precisely why it yields Jacobi and not Gauss-Seidel.
fn jacobi_mttkrp_all<T>(
    tensor: &DenseND<T>,
    factors: &[Array2<T>],
) -> Result<Vec<Array2<T>>, CpError>
where
    T: Float + Send + Sync + 'static,
{
    let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();
    let tree = tenrso_kernels::DimTree::new(tensor.shape())
        .map_err(|e| CpError::ShapeMismatch(e.to_string()))?;

    #[cfg(feature = "parallel")]
    let result = tree.mttkrp_all_parallel(&tensor.view(), &factor_views);
    #[cfg(not(feature = "parallel"))]
    let result = tree.mttkrp_all(&tensor.view(), &factor_views);

    result.map_err(|e| CpError::ShapeMismatch(e.to_string()))
}

/// One **stabilized Jacobi sweep**: all `N` factors solved independently against the
/// frozen factor set, then column-normalized, then re-weighted by an optimal
/// least-squares refit of the `R` component weights.
///
/// # Why the normalization + weight refit is not optional
///
/// The *naked* simultaneous update `A_k ← M_k · G_k^{-1}` for all `k` at once —
/// i.e. exactly what you get by dropping [`tenrso_kernels::DimTree::mttkrp_all`]
/// into an ALS loop — **provably diverges**. Take the rank-1, order-`N` case with
/// unit-norm true factors `a_k` and an iterate `A_k = α_k a_k`. Writing
/// `p = Π_k α_k` for the reconstruction scale, the exact per-mode solve gives
/// `α_k' = 1 / Π_{j≠k} α_j = α_k / p`, hence
///
/// ```text
/// p' = Π_k α_k' = p / p^N = p^{1-N}
/// ```
///
/// The fixed point `p = 1` is correct, but linearizing `p = 1 + ε` gives
/// `p' ≈ 1 - (N-1)·ε`: the scale error is **amplified by `N-1` every sweep**
/// (doubling for a 3-way tensor, tripling for a 4-way one) and alternates sign. The
/// iteration blows up. Gauss-Seidel does not have this mode, because each factor is
/// solved against partners that already absorbed the current sweep's rescaling.
///
/// Normalizing every column to unit norm collapses that unstable direction into the
/// `R` explicit weights `λ`, which we then choose *optimally* rather than
/// multiplicatively: with `U_k` the normalized factors,
///
/// ```text
/// λ = H^{-1} g,   H = ⊛_k (U_kᵀ U_k),   g_r = <X, u_{0r} ∘ … ∘ u_{N-1,r}>
/// ```
///
/// which is the exact minimizer of `‖X - Σ_r λ_r u_{0r} ∘ … ∘ u_{N-1,r}‖²` over `λ`.
/// The scale subspace is therefore *solved*, not iterated, and the divergence is
/// gone. `g` comes from the mode-0 MTTKRP of the normalized factors, which the fit
/// needs anyway — so the stabilization costs `O(R³ + I_0·R)` on top of the MTTKRP.
///
/// # Returns
///
/// `(factors, mttkrp_0)` where `factors[0]` has absorbed `λ` and `mttkrp_0` is the
/// mode-0 MTTKRP of the returned set (it does not depend on factor 0, so absorbing
/// `λ` there leaves it valid).
fn jacobi_sweep<T>(
    tensor: &DenseND<T>,
    tree: &AlsDimTree,
    tensor_view: &scirs2_core::ndarray_ext::ArrayView<T, scirs2_core::ndarray_ext::IxDyn>,
    factors: &[Array2<T>],
) -> Result<(Vec<Array2<T>>, Array2<T>), CpError>
where
    T: Float + NumAssign + Sum + scirs2_core::ndarray_ext::ScalarOperand + Send + Sync + 'static,
{
    let n_modes = factors.len();
    let rank = factors[0].shape()[1];

    // 1. All N MTTKRPs against ONE frozen snapshot — the dimension tree's natural
    //    (Jacobi) mode, one shared pass over the tensor.
    let mttkrps = jacobi_mttkrp_all(tensor, factors)?;

    // 2. Independent solves. Every Gram is also taken against the frozen snapshot;
    //    that is what makes this Jacobi rather than a half-updated hybrid.
    let mut updated = Vec::with_capacity(n_modes);
    for (mode, mttkrp_result) in mttkrps.iter().enumerate() {
        let gram = compute_gram_hadamard(factors, mode);
        updated.push(solve_least_squares(mttkrp_result, &gram)?);
    }

    // 3. Column-normalize, pushing all scale out of the factors. A zero column stays
    //    zero and simply receives weight 0 in step 5.
    for factor in &mut updated {
        for r in 0..rank {
            let mut norm_sq = T::zero();
            for i in 0..factor.shape()[0] {
                norm_sq += factor[[i, r]] * factor[[i, r]];
            }
            let norm = norm_sq.sqrt();
            if norm > T::zero() {
                for i in 0..factor.shape()[0] {
                    factor[[i, r]] /= norm;
                }
            }
        }
    }

    // 4. Mode-0 MTTKRP of the normalized factors. Independent of `updated[0]`, so it
    //    survives step 6 and doubles as the fit's inner product.
    let mttkrp_0 = tree.mttkrp_mode(tensor_view, &updated, 0)?;

    // 5. Optimal weights: lambda = H^{-1} g, the exact least-squares minimizer over
    //    the R-dimensional scale subspace.
    let mut hadamard_gram = Array2::<T>::ones((rank, rank));
    for factor in &updated {
        let gram = compute_gram_matrix(factor);
        for r1 in 0..rank {
            for r2 in 0..rank {
                hadamard_gram[[r1, r2]] *= gram[[r1, r2]];
            }
        }
    }
    let mut rhs = Array2::<T>::zeros((1, rank));
    for r in 0..rank {
        let mut acc = T::zero();
        for i in 0..updated[0].shape()[0] {
            acc += mttkrp_0[[i, r]] * updated[0][[i, r]];
        }
        rhs[[0, r]] = acc;
    }
    let lambda = solve_least_squares(&rhs, &hadamard_gram)?;

    // 6. Absorb the weights into factor 0, keeping `CpDecomp::weights` free-standing
    //    (the model is unchanged: [[A_0 diag(lambda), U_1, ..., U_{N-1}]]).
    for r in 0..rank {
        let weight = lambda[[0, r]];
        for i in 0..updated[0].shape()[0] {
            updated[0][[i, r]] *= weight;
        }
    }

    Ok((updated, mttkrp_0))
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
/// Like [`cp_als`], this is **Gauss-Seidel**: the projection/regularization applied
/// to mode `k` is visible to modes `k+1..N-1` within the same sweep. It runs on the
/// same Gauss-Seidel-preserving dimension tree (see the internal `dimtree` module); the
/// constraint steps are simply part of the per-mode update the tree calls back into.
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
/// # Errors
///
/// As [`cp_als`], plus invalid regularization parameters or a negative entry in the
/// input tensor when non-negativity is requested with input validation on.
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
    let tol_t: T = validate_cp_args(tensor, rank, tol)?;
    let n_modes = tensor.rank();

    constraints.regularization.validate()?;

    // Legacy l2_reg validation (when not using advanced regularization)
    if matches!(constraints.regularization, RegularizationType::None) && constraints.l2_reg < 0.0 {
        return Err(CpError::InvalidTolerance(constraints.l2_reg));
    }

    // Validate non-negative input tensor when constraint is active
    if constraints.nonnegative && constraints.validate_nonneg_input {
        validate_nonnegative(tensor)?;
    }

    let mut factors = initialize_factors(tensor, rank, init)?;

    // Apply initial constraints
    if constraints.nonnegative {
        for factor in &mut factors {
            factor.mapv_inplace(|x| x.max(T::zero()));
        }
    }

    let tensor_norm_sq = compute_norm_squared(tensor);
    let tensor_view = tensor.view();
    let tree = AlsDimTree::new(tensor.shape())?;

    // Effective regularization parameters (hoisted: constant across sweeps).
    let l2_lambda = constraints.effective_l2();
    let l1_lambda = constraints.effective_l1();

    let mut tracker = ConvergenceTracker::<T>::new(max_iters);
    let start_time = std::time::Instant::now();

    for iter in 0..max_iters {
        if let Some(limit) = time_limit {
            if start_time.elapsed() > limit {
                tracker.reason = ConvergenceReason::TimeLimit;
                break;
            }
        }

        tracker.iters = iter + 1;

        let mut update = |mode: usize,
                          mttkrp_result: &Array2<T>,
                          factors: &mut Vec<Array2<T>>|
         -> Result<(), CpError> {
            // Step 1: Hadamard product of the Gram matrices.
            let mut gram = compute_gram_hadamard(factors, mode);

            // Step 2: Regularize the Gram matrix.
            match constraints.regularization {
                RegularizationType::Tikhonov { lambda, order } if order > 0 => {
                    let lambda_t: T = cast_f64(lambda, "tikhonov lambda")?;
                    apply_tikhonov_to_gram(&mut gram, lambda_t, order);
                }
                _ => {
                    if l2_lambda > 0.0 {
                        let reg: T = cast_f64(l2_lambda, "l2_lambda")?;
                        for i in 0..gram.nrows() {
                            gram[[i, i]] += reg;
                        }
                    }
                }
            }

            // Step 3: Solve least squares.
            factors[mode] = solve_least_squares(mttkrp_result, &gram)?;

            // Step 4: L1 regularization (soft-thresholding).
            if l1_lambda > 0.0 {
                let threshold: T = cast_f64(l1_lambda, "l1_lambda")?;
                soft_threshold(&mut factors[mode], threshold);
            }

            // Step 5: Constraints. These land in `factors[mode]` *before* the tree
            // descends to modes `mode+1..`, so later modes in this same sweep see
            // the projected factor — exactly as in the original per-mode loop.
            if constraints.nonnegative {
                factors[mode].mapv_inplace(|x| x.max(T::zero()));
            }

            if constraints.orthogonal {
                factors[mode] = orthonormalize_factor(&factors[mode])?;
            }

            Ok(())
        };

        let last_mttkrp = tree.gauss_seidel_sweep(&tensor_view, &mut factors, &mut update)?;

        // Mode N-1 is updated last, so `last_mttkrp` was taken against the final
        // (already projected/regularized) factors 0..N-2 and yields <X, [[A]]> for
        // the post-sweep factor set.
        let fit = compute_fit_from_mttkrp(&factors, &last_mttkrp, n_modes - 1, tensor_norm_sq);

        if tracker.record(iter, fit, tol_t) {
            break;
        }
    }

    Ok(tracker.finish(factors))
}
