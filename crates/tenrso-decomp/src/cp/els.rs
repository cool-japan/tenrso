//! **Enhanced Line Search (ELS)** for CP-ALS — an *exact*, tensor-pass-free line
//! search along the joint extrapolation direction of a whole ALS sweep.
//!
//! # Why the search has to be joint
//!
//! Searching along a *single mode's* ALS direction is vacuous: with the other factors
//! frozen, the objective restricted to factor `k` is the quadratic
//!
//! ```text
//! f(A) = ‖X‖² − 2·⟨M_k, A⟩ + ⟨G_k, AᵀA⟩,      ∇f = 0  ⟺  A = M_k · G_k⁻¹
//! ```
//!
//! whose global minimiser is *exactly* the ALS solve. The optimal step along the line
//! through it is therefore `α = 1` identically — always, for every tensor. There is
//! nothing to search for.
//!
//! The moment all `N` factors move together, that stops being true. Let a Gauss-Seidel
//! sweep carry `A_k^prev → A_k^new`, put `D_k = A_k^new − A_k^prev`, and search the line
//!
//! ```text
//! A_k(α) = A_k^prev + α · D_k        (all k simultaneously)
//! ```
//!
//! `α = 1` reproduces the sweep, `α > 1` extrapolates (Rajih–Comon–Harshman ELS; this
//! is what breaks CP-ALS *swamps*), and `α < 1` **damps** — which the joint direction
//! genuinely needs, because only mode `N−1` is optimal for the post-sweep factor set;
//! modes `0..N−2` were solved against partners that moved afterwards, so the joint move
//! can overshoot. A line search that cannot return `α < 1` cannot do its job.
//!
//! # Why it costs no extra tensor passes to *evaluate*
//!
//! Along that line the squared error is a **polynomial in `α` of degree `2N`**, and its
//! coefficients can be built once per sweep:
//!
//! ```text
//! ‖X − X̂(α)‖² = ‖X‖² − 2·g(α) + h(α)
//!
//!   g(α) = ⟨X, X̂(α)⟩            — degree N
//!   h(α) = ‖X̂(α)‖²              — degree 2N
//! ```
//!
//! * **`h`** needs *no tensor at all*. `A_k(α)ᵀA_k(α) = P_k + α·S_k + α²·Q_k` with
//!   `P_k = A_kᵀA_k`, `S_k = A_kᵀD_k + D_kᵀA_k`, `Q_k = D_kᵀD_k` (all `R×R`), and
//!   `h(α) = Σ_{r,s} Π_k (P_k + αS_k + α²Q_k)[r,s]` — a product of `N` quadratics,
//!   `O(Σ_k I_k·R² + N²·R²)` flops.
//!
//! * **`g`** is routed through the mode-0 MTTKRP:
//!   `g(α) = ⟨M(α), A_0(α)⟩` where `M(α) = MTTKRP(X; A_1(α), …, A_{N−1}(α))` is a
//!   *matrix polynomial of degree `N−1`*. We recover its `N` monomial coefficients
//!   `M_0 … M_{N−1}` by evaluating `M` at `N` distinct nodes and Lagrange-interpolating.
//!   **Node `α = 0` is free** — it is exactly the mode-0 leaf the Gauss-Seidel sweep
//!   already computed. So the whole polynomial costs `N−1` extra MTTKRPs and nothing
//!   else.
//!
//! Once the `2N+1` coefficients exist, `‖X − X̂(α)‖²` at *any* `α` is a Horner
//! evaluation — `O(N)` flops. The search is then exact and effectively free: we root
//! the derivative (degree `2N−1`) by scan + bisection and take the global minimiser on
//! `[α_min, α_max]`.
//!
//! # Cost, honestly
//!
//! Per sweep: `2·nnz·R` (the Gauss-Seidel dimension-tree sweep) `+ (N−1)·nnz·R` (the ELS
//! probes) `= (N+1)·nnz·R`, versus `2·nnz·R` for plain [`cp_als`](crate::cp_als). For a
//! 3-way tensor that is **2× the per-sweep cost**. ELS therefore has to cut the sweep
//! count by more than 2× to pay for itself — it does that in swamps and does not on easy
//! problems. See the `cp_els_bench` example and the docs on
//! [`cp_als_accelerated`](super::cp_als_accelerated) for the measured numbers.
//!
//! The fit comes back **free**: `p(α*)` *is* `‖X − X̂(α*)‖²`, so the accelerated driver
//! needs no fit pass at all.
//!
//! # SciRS2 Integration
//!
//! All array operations go through `scirs2_core::ndarray_ext`. Direct `ndarray`/`rand`
//! use is forbidden per SCIRS2_INTEGRATION_POLICY.md.

use super::dimtree::AlsDimTree;
use super::types::CpError;
use scirs2_core::ndarray_ext::{Array2, ArrayView, IxDyn};
use scirs2_core::numeric::{Float, NumCast, ToPrimitive};

/// Lower bound of the ELS search interval.
///
/// `0` means "do not move at all". It is never *strictly* better than `α = 1` (an exact
/// Gauss-Seidel sweep cannot increase the error), so including it costs nothing and
/// makes the interval closed under damping.
pub(crate) const ELS_ALPHA_MIN: f64 = 0.0;

/// Upper bound of the ELS search interval.
///
/// The error polynomial has even degree `2N` with a non-negative leading coefficient
/// (`Σ_{r,s} Π_k Q_k[r,s]`, a sum of entries of a Hadamard product of Gram matrices —
/// PSD by the Schur product theorem), so its global minimiser is finite and no bound is
/// needed for *correctness*. The bound is a **conditioning** guard: a huge `α` inflates
/// the factor norms and the next sweep's Gram matrices along with them. `4` lets ELS take
/// a step four times the ALS step — enough for the swamp-breaking regime, which is where
/// the measured wins come from — without letting a near-flat polynomial fling the iterate
/// into a badly scaled corner.
pub(crate) const ELS_ALPHA_MAX: f64 = 4.0;

/// Bisection steps used to isolate each root of `p'`.
///
/// Each step halves the bracket, so 40 steps take a bracket of width `≤ 4` down to
/// `≈ 4·2⁻⁴⁰ ≈ 4e-12` — below the precision at which `α` could matter. Every step is a
/// single Horner evaluation of a degree-`2N−1` polynomial (tens of flops), so the whole
/// search is free next to one MTTKRP.
pub(crate) const ELS_REFINE_ITERS: usize = 40;

/// Convert `T` to `f64`, with a typed error instead of a panic.
#[inline]
fn as_f64<T: ToPrimitive>(value: T) -> Result<f64, CpError> {
    value.to_f64().ok_or_else(|| {
        CpError::ShapeMismatch("factor entry is not representable as f64".to_string())
    })
}

/// Copy an `Array2<T>` into an `Array2<f64>`.
///
/// The ELS polynomial is assembled in `f64` regardless of `T`: it is only `2N+1`
/// numbers, and `‖X‖² − 2g + h` is a difference of large, nearly equal quantities near
/// convergence. Doing that cancellation in `f32` would destroy the very digits the
/// search depends on.
fn to_f64_matrix<T: ToPrimitive + Copy>(matrix: &Array2<T>) -> Result<Array2<f64>, CpError> {
    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
    let mut out = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = as_f64(matrix[[i, j]])?;
        }
    }
    Ok(out)
}

/// Frobenius inner product `Σ_{i,j} a[i,j]·b[i,j]`.
#[inline]
fn frobenius_dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.shape()[0] {
        for j in 0..a.shape()[1] {
            acc += a[[i, j]] * b[[i, j]];
        }
    }
    acc
}

/// Monomial coefficients of the Lagrange basis for `nodes`.
///
/// Returns `c` with `c[j][d]` the coefficient of `αᵈ` in
/// `L_j(α) = Π_{m≠j} (α − t_m) / (t_j − t_m)`, so that a polynomial known at the nodes
/// is recovered in the monomial basis by `p_d = Σ_j c[j][d] · p(t_j)`.
///
/// # Errors
///
/// The nodes must be pairwise distinct (otherwise a Lagrange denominator vanishes).
fn lagrange_monomial_coeffs(nodes: &[f64]) -> Result<Vec<Vec<f64>>, CpError> {
    let n = nodes.len();
    let mut basis = Vec::with_capacity(n);

    for (j, &t_j) in nodes.iter().enumerate() {
        // Numerator Π_{m≠j} (α − t_m), built up one linear factor at a time.
        let mut poly = vec![0.0f64; n];
        poly[0] = 1.0;
        let mut degree = 0usize;
        let mut denominator = 1.0f64;

        for (m, &t_m) in nodes.iter().enumerate() {
            if m == j {
                continue;
            }
            // poly ← poly · (α − t_m), descending so we never clobber poly[d-1].
            for d in (0..=degree + 1).rev() {
                let lower = if d >= 1 { poly[d - 1] } else { 0.0 };
                let current = if d <= degree { poly[d] } else { 0.0 };
                poly[d] = lower - t_m * current;
            }
            degree += 1;
            denominator *= t_j - t_m;
        }

        if denominator == 0.0 || !denominator.is_finite() {
            return Err(CpError::ShapeMismatch(
                "ELS interpolation nodes must be pairwise distinct".to_string(),
            ));
        }
        for coefficient in &mut poly {
            *coefficient /= denominator;
        }
        basis.push(poly);
    }

    Ok(basis)
}

/// Evaluate a polynomial given in the monomial basis, by Horner's rule.
#[inline]
fn horner(coeffs: &[f64], alpha: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * alpha + c;
    }
    acc
}

/// Coefficients of `p'` given the coefficients of `p`.
fn derivative(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.len() <= 1 {
        return vec![0.0];
    }
    (1..coeffs.len()).map(|d| coeffs[d] * (d as f64)).collect()
}

/// The exact squared reconstruction error along the ELS line, as a polynomial in `α`.
///
/// `self.coeffs[d]` multiplies `αᵈ`; the degree is `2N`. Evaluating it at `α` yields
/// `‖X − [[A_0 + αD_0, …, A_{N−1} + αD_{N−1}]]‖²` **exactly** (up to rounding) — no
/// reconstruction, no tensor pass.
#[derive(Debug, Clone)]
pub(crate) struct ErrorPolynomial {
    coeffs: Vec<f64>,
}

impl ErrorPolynomial {
    /// Build the polynomial for the sweep `prev_factors → new_factors`.
    ///
    /// `mttkrp_0_prev` must be the mode-0 MTTKRP taken against `prev_factors` — which is
    /// exactly the first leaf a Gauss-Seidel sweep computes, so the caller gets it for
    /// free. Passing anything else silently produces the wrong polynomial, hence the
    /// `pub(crate)` visibility and the debug assertion in the tests.
    ///
    /// # Complexity
    ///
    /// `(N−1)·nnz·R` (the interpolation probes) `+ O(Σ_k I_k·R² + N²·R²)`.
    ///
    /// # Errors
    ///
    /// Shape mismatches between the factor sets, an order-`< 2` tensor, or an MTTKRP
    /// failure from the dimension tree.
    pub(crate) fn build<T>(
        tensor: &ArrayView<T, IxDyn>,
        tree: &AlsDimTree,
        prev_factors: &[Array2<T>],
        new_factors: &[Array2<T>],
        mttkrp_0_prev: &Array2<T>,
        tensor_norm_sq: f64,
    ) -> Result<Self, CpError>
    where
        T: Float + Send + Sync + 'static,
    {
        let n_modes = prev_factors.len();
        if n_modes < 2 {
            return Err(CpError::ShapeMismatch(format!(
                "ELS requires a tensor of order >= 2 (got order {n_modes})"
            )));
        }
        if new_factors.len() != n_modes {
            return Err(CpError::ShapeMismatch(format!(
                "ELS factor sets disagree on order: {} vs {n_modes}",
                new_factors.len()
            )));
        }
        let rank = prev_factors[0].shape()[1];
        for mode in 0..n_modes {
            if prev_factors[mode].shape() != new_factors[mode].shape() {
                return Err(CpError::ShapeMismatch(format!(
                    "ELS factor sets disagree on mode-{mode} shape: {:?} vs {:?}",
                    prev_factors[mode].shape(),
                    new_factors[mode].shape()
                )));
            }
        }

        // ── The search direction ────────────────────────────────────────────────
        let diffs: Vec<Array2<T>> = (0..n_modes)
            .map(|mode| {
                let rows = prev_factors[mode].shape()[0];
                Array2::<T>::from_shape_fn((rows, rank), |(i, r)| {
                    new_factors[mode][[i, r]] - prev_factors[mode][[i, r]]
                })
            })
            .collect();

        // ── g(α) = ⟨X, X̂(α)⟩, via the mode-0 matrix polynomial M(α) ──────────────
        //
        // M(α) = MTTKRP(X; A_1(α), …, A_{N-1}(α)) has degree N-1 in α, so N nodes pin it
        // down exactly. Node t_0 = 0 is the sweep's own mode-0 leaf: free.
        let nodes: Vec<f64> = (0..n_modes).map(|j| j as f64).collect();
        let lagrange = lagrange_monomial_coeffs(&nodes)?;

        let mut probes: Vec<Array2<f64>> = Vec::with_capacity(n_modes);
        probes.push(to_f64_matrix(mttkrp_0_prev)?);
        for &node in nodes.iter().skip(1) {
            let node_t: T = NumCast::from(node).ok_or_else(|| {
                CpError::ShapeMismatch(format!("ELS node {node} is not representable in T"))
            })?;
            let mut probe_factors: Vec<Array2<T>> = Vec::with_capacity(n_modes);
            // A mode-0 MTTKRP ignores factor 0; the slot only has to have the right shape.
            probe_factors.push(prev_factors[0].clone());
            for mode in 1..n_modes {
                let rows = prev_factors[mode].shape()[0];
                probe_factors.push(Array2::<T>::from_shape_fn((rows, rank), |(i, r)| {
                    prev_factors[mode][[i, r]] + node_t * diffs[mode][[i, r]]
                }));
            }
            probes.push(to_f64_matrix(&tree.mttkrp_mode(
                tensor,
                &probe_factors,
                0,
            )?)?);
        }

        // M_d = Σ_j L_j,d · M(t_j)
        let rows_0 = prev_factors[0].shape()[0];
        let mut m_mono: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((rows_0, rank)); n_modes];
        for (j, probe) in probes.iter().enumerate() {
            for (d, target) in m_mono.iter_mut().enumerate() {
                let weight = lagrange[j][d];
                if weight == 0.0 {
                    continue;
                }
                for i in 0..rows_0 {
                    for r in 0..rank {
                        target[[i, r]] += weight * probe[[i, r]];
                    }
                }
            }
        }

        // g(α) = ⟨M(α), A_0 + α·D_0⟩  ⇒  g_d = ⟨M_d, A_0⟩ + ⟨M_{d-1}, D_0⟩.
        let a0 = to_f64_matrix(&prev_factors[0])?;
        let d0 = to_f64_matrix(&diffs[0])?;
        let mut g = vec![0.0f64; n_modes + 1];
        for (d, m_d) in m_mono.iter().enumerate() {
            g[d] += frobenius_dot(m_d, &a0);
            g[d + 1] += frobenius_dot(m_d, &d0);
        }

        // ── h(α) = ‖X̂(α)‖², from the R×R Gram blocks only. No tensor. ────────────
        let mut gram_p = Vec::with_capacity(n_modes);
        let mut gram_s = Vec::with_capacity(n_modes);
        let mut gram_q = Vec::with_capacity(n_modes);
        for mode in 0..n_modes {
            let a = to_f64_matrix(&prev_factors[mode])?;
            let d = to_f64_matrix(&diffs[mode])?;
            let p = a.t().dot(&a);
            let cross = a.t().dot(&d);
            let s =
                Array2::<f64>::from_shape_fn((rank, rank), |(r, c)| cross[[r, c]] + cross[[c, r]]);
            let q = d.t().dot(&d);
            gram_p.push(p);
            gram_s.push(s);
            gram_q.push(q);
        }

        let degree = 2 * n_modes;
        let mut coeffs = vec![0.0f64; degree + 1];
        // Scratch for the running product of the N quadratics, reused across (r, s).
        let mut current = vec![0.0f64; degree + 1];
        let mut next = vec![0.0f64; degree + 1];
        for r in 0..rank {
            for s in 0..rank {
                current[0] = 1.0;
                for slot in current.iter_mut().take(degree + 1).skip(1) {
                    *slot = 0.0;
                }
                let mut current_degree = 0usize;

                for mode in 0..n_modes {
                    let (p, sc, q) = (
                        gram_p[mode][[r, s]],
                        gram_s[mode][[r, s]],
                        gram_q[mode][[r, s]],
                    );
                    for slot in next.iter_mut().take(current_degree + 3) {
                        *slot = 0.0;
                    }
                    for d in 0..=current_degree {
                        let c = current[d];
                        if c == 0.0 {
                            continue;
                        }
                        next[d] += c * p;
                        next[d + 1] += c * sc;
                        next[d + 2] += c * q;
                    }
                    current_degree += 2;
                    current[..=current_degree].copy_from_slice(&next[..=current_degree]);
                }

                for (d, slot) in coeffs.iter_mut().enumerate() {
                    *slot += current[d];
                }
            }
        }

        // ── p(α) = ‖X‖² − 2·g(α) + h(α) ─────────────────────────────────────────
        coeffs[0] += tensor_norm_sq;
        for (d, &g_d) in g.iter().enumerate() {
            coeffs[d] -= 2.0 * g_d;
        }

        Ok(Self { coeffs })
    }

    /// `‖X − X̂(α)‖²` at `α`.
    #[inline]
    pub(crate) fn eval(&self, alpha: f64) -> f64 {
        horner(&self.coeffs, alpha)
    }

    /// Globally minimise the polynomial on `[lo, hi]`.
    ///
    /// Returns `(α*, p(α*))`. The candidate set is `{lo, hi, 1}` plus every real root of
    /// `p'` in `[lo, hi]`, each isolated by a sign-change scan and `refine_iters`
    /// bisection steps. Including `α = 1` unconditionally is what makes the accelerated
    /// driver **never worse than a plain ALS sweep**: `p(α*) ≤ p(1) ≤ p(0)`, the second
    /// inequality because an exact Gauss-Seidel sweep cannot increase the error.
    ///
    /// The scan uses `16·(2N+1)` sub-intervals, comfortably more than the `2N−1` roots a
    /// degree-`2N−1` polynomial can have; each probe is one Horner evaluation.
    ///
    /// A non-finite polynomial (a factor blew up to `inf`/`NaN`) degrades to `α = 1`,
    /// i.e. to exactly the plain ALS sweep, rather than propagating garbage.
    pub(crate) fn minimize(&self, lo: f64, hi: f64, refine_iters: usize) -> (f64, f64) {
        if !self.coeffs.iter().all(|c| c.is_finite()) || lo >= hi {
            return (1.0, self.eval(1.0));
        }

        let deriv = derivative(&self.coeffs);

        let mut best_alpha = 1.0f64.clamp(lo, hi);
        let mut best_value = self.eval(best_alpha);
        let consider = |alpha: f64, best_alpha: &mut f64, best_value: &mut f64| {
            let value = self.eval(alpha);
            if value.is_finite() && value < *best_value {
                *best_value = value;
                *best_alpha = alpha;
            }
        };
        consider(lo, &mut best_alpha, &mut best_value);
        consider(hi, &mut best_alpha, &mut best_value);

        let scan_points = 16 * self.coeffs.len();
        let mut left = lo;
        let mut left_deriv = horner(&deriv, left);
        for step in 1..=scan_points {
            let right = lo + (hi - lo) * (step as f64) / (scan_points as f64);
            let right_deriv = horner(&deriv, right);

            if left_deriv * right_deriv <= 0.0 {
                // A root of p' lies in [left, right] (or sits on an endpoint). Take the
                // endpoints too, so a root exactly at a grid point is never missed.
                consider(left, &mut best_alpha, &mut best_value);
                consider(right, &mut best_alpha, &mut best_value);

                let (mut a, mut b) = (left, right);
                let mut fa = left_deriv;
                for _ in 0..refine_iters {
                    let mid = 0.5 * (a + b);
                    let fm = horner(&deriv, mid);
                    if fm == 0.0 {
                        a = mid;
                        b = mid;
                        break;
                    }
                    if (fa < 0.0) == (fm < 0.0) {
                        a = mid;
                        fa = fm;
                    } else {
                        b = mid;
                    }
                }
                consider(0.5 * (a + b), &mut best_alpha, &mut best_value);
            }

            left = right;
            left_deriv = right_deriv;
        }

        (best_alpha, best_value)
    }
}

#[cfg(test)]
mod tests {
    use super::super::helpers::{
        compute_fit, compute_gram_hadamard, compute_norm_squared, initialize_factors,
        solve_least_squares,
    };
    use super::super::types::InitStrategy;
    use super::*;
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    use tenrso_core::DenseND;

    /// Deterministic data generator — the tests must not depend on a thread RNG.
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

    fn random_tensor(shape: &[usize], seed: u64) -> DenseND<f64> {
        let mut rng = Lcg::new(seed);
        let numel: usize = shape.iter().product();
        let values: Vec<f64> = (0..numel).map(|_| rng.normal()).collect();
        DenseND::from_array(
            Array::<f64, IxDyn>::from_shape_vec(IxDyn(shape), values)
                .expect("value count matches shape"),
        )
    }

    /// Explicit `‖X − [[A(α)]]‖²`, built the slow honest way (full reconstruction) so the
    /// polynomial has something independent to be checked against.
    fn brute_force_error_sq(
        tensor: &DenseND<f64>,
        prev: &[Array2<f64>],
        new: &[Array2<f64>],
        alpha: f64,
    ) -> f64 {
        let blended: Vec<Array2<f64>> = (0..prev.len())
            .map(|mode| {
                let rows = prev[mode].shape()[0];
                let rank = prev[mode].shape()[1];
                Array2::<f64>::from_shape_fn((rows, rank), |(i, r)| {
                    prev[mode][[i, r]] + alpha * (new[mode][[i, r]] - prev[mode][[i, r]])
                })
            })
            .collect();
        let views: Vec<_> = blended.iter().map(|f| f.view()).collect();
        let recon = tenrso_kernels::cp_reconstruct(&views, None).expect("reconstruct");
        let mut acc = 0.0;
        for (&x, &y) in tensor.view().iter().zip(recon.iter()) {
            acc += (x - y) * (x - y);
        }
        acc
    }

    /// Run one Gauss-Seidel sweep, returning `(prev, new, mode-0 MTTKRP of prev)`.
    fn one_sweep(
        tensor: &DenseND<f64>,
        tree: &AlsDimTree,
        factors: &[Array2<f64>],
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>, Array2<f64>) {
        let prev = factors.to_vec();
        let mut current = factors.to_vec();
        let mut mttkrp_0: Option<Array2<f64>> = None;
        {
            let mut update =
                |mode: usize, m: &Array2<f64>, f: &mut Vec<Array2<f64>>| -> Result<(), CpError> {
                    if mode == 0 {
                        mttkrp_0 = Some(m.clone());
                    }
                    let gram = compute_gram_hadamard(f, mode);
                    f[mode] = solve_least_squares(m, &gram)?;
                    Ok(())
                };
            tree.gauss_seidel_sweep(&tensor.view(), &mut current, &mut update)
                .expect("sweep");
        }
        (prev, current, mttkrp_0.expect("mode 0 is visited first"))
    }

    #[test]
    fn polynomial_matches_brute_force_reconstruction_error() {
        for (shape, rank, seed) in [
            (vec![6usize, 5, 7], 3usize, 1u64),
            (vec![5, 5, 4, 3], 2, 2),
            (vec![9, 8], 4, 3),
        ] {
            let tensor = random_tensor(&shape, seed);
            let tree = AlsDimTree::new(&shape).expect("tree");
            let factors =
                initialize_factors(&tensor, rank, InitStrategy::Svd).expect("init factors");
            let (prev, new, mttkrp_0) = one_sweep(&tensor, &tree, &factors);

            let norm_sq = compute_norm_squared(&tensor);
            let poly =
                ErrorPolynomial::build(&tensor.view(), &tree, &prev, &new, &mttkrp_0, norm_sq)
                    .expect("polynomial");

            // Degree must be exactly 2N.
            assert_eq!(poly.coeffs.len(), 2 * shape.len() + 1);

            for &alpha in &[-0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.7] {
                let exact = brute_force_error_sq(&tensor, &prev, &new, alpha);
                let from_poly = poly.eval(alpha);
                let scale = exact.abs().max(1.0);
                assert!(
                    (exact - from_poly).abs() / scale < 1e-9,
                    "shape {shape:?} alpha {alpha}: polynomial {from_poly:.12e} != \
                     brute force {exact:.12e}"
                );
            }
        }
    }

    /// The whole point of the polynomial: `p(1)` is the post-sweep error, and it must
    /// agree with what `compute_fit` says about the post-sweep factors — while costing
    /// no reconstruction.
    #[test]
    fn polynomial_at_one_reproduces_the_post_sweep_fit() {
        let shape = [8usize, 7, 6];
        let tensor = random_tensor(&shape, 42);
        let tree = AlsDimTree::new(&shape).expect("tree");
        let factors = initialize_factors(&tensor, 4, InitStrategy::Svd).expect("init");
        let (prev, new, mttkrp_0) = one_sweep(&tensor, &tree, &factors);

        let norm_sq = compute_norm_squared(&tensor);
        let poly = ErrorPolynomial::build(&tensor.view(), &tree, &prev, &new, &mttkrp_0, norm_sq)
            .expect("polynomial");

        let fit_from_poly = 1.0 - poly.eval(1.0).max(0.0).sqrt() / norm_sq.sqrt();
        let fit_direct = compute_fit(&tensor, &new, norm_sq).expect("fit");
        assert!(
            (fit_from_poly - fit_direct).abs() < 1e-10,
            "fit from polynomial {fit_from_poly} != fit from reconstruction {fit_direct}"
        );

        // And the sweep really did decrease the error: p(1) <= p(0).
        assert!(poly.eval(1.0) <= poly.eval(0.0) + 1e-12);
    }

    /// `minimize` must find the *global* minimum on the interval, including when it lies
    /// strictly below 1 — the case the old per-mode search was structurally unable to
    /// express.
    #[test]
    fn minimize_recovers_the_global_minimum_of_a_known_polynomial() {
        // p(α) = (α − 0.3)² · (α − 2.5)² + 0.1·(α − 0.3)²  →  a clear global min near 0.3,
        // a second local min near 2.5. Expand to monomial form.
        // (α−a)²(α−b)² = ((α−a)(α−b))² with (α−a)(α−b) = α² − (a+b)α + ab.
        let (a, b) = (0.3f64, 2.5f64);
        let q = [a * b, -(a + b), 1.0]; // q(α) = α² − (a+b)α + ab
        let mut coeffs = vec![0.0f64; 5];
        for i in 0..3 {
            for j in 0..3 {
                coeffs[i + j] += q[i] * q[j];
            }
        }
        // + 0.1·(α − a)²
        coeffs[0] += 0.1 * a * a;
        coeffs[1] += 0.1 * (-2.0 * a);
        coeffs[2] += 0.1;

        let poly = ErrorPolynomial { coeffs };
        let (alpha, value) = poly.minimize(0.0, 4.0, ELS_REFINE_ITERS);
        assert!(
            (alpha - 0.3).abs() < 1e-6,
            "expected the global minimiser near 0.3, got {alpha}"
        );
        assert!(value <= poly.eval(1.0));
        assert!(alpha < 1.0, "the search must be able to return alpha < 1");
    }

    /// The polynomial is the exact error, so a search that returns `α*` must have
    /// `p(α*) ≤ p(1)`: ELS can never be worse than the plain sweep it accelerates.
    #[test]
    fn searched_optimum_is_never_worse_than_the_plain_als_step() {
        let shape = [10usize, 9, 8];
        let tensor = random_tensor(&shape, 7);
        let tree = AlsDimTree::new(&shape).expect("tree");
        let mut factors = initialize_factors(&tensor, 5, InitStrategy::Random).expect("init");
        let norm_sq = compute_norm_squared(&tensor);

        for _ in 0..15 {
            let (prev, new, mttkrp_0) = one_sweep(&tensor, &tree, &factors);
            let poly =
                ErrorPolynomial::build(&tensor.view(), &tree, &prev, &new, &mttkrp_0, norm_sq)
                    .expect("polynomial");
            let (alpha, value) = poly.minimize(ELS_ALPHA_MIN, ELS_ALPHA_MAX, ELS_REFINE_ITERS);

            assert!(
                value <= poly.eval(1.0) + 1e-12,
                "alpha* = {alpha} gave error {value} > plain ALS error {}",
                poly.eval(1.0)
            );
            assert!((ELS_ALPHA_MIN..=ELS_ALPHA_MAX).contains(&alpha));

            // Walk to the searched point, exactly as the driver does.
            for mode in 0..shape.len() {
                let rows = shape[mode];
                factors[mode] = Array2::<f64>::from_shape_fn((rows, 5), |(i, r)| {
                    prev[mode][[i, r]] + alpha * (new[mode][[i, r]] - prev[mode][[i, r]])
                });
            }
        }
    }

    #[test]
    fn lagrange_basis_is_a_partition_of_unity() {
        let nodes = [0.0, 1.0, 2.0, 3.0];
        let basis = lagrange_monomial_coeffs(&nodes).expect("basis");
        // L_j(t_m) = δ_{jm}
        for (j, coeffs) in basis.iter().enumerate() {
            for (m, &t_m) in nodes.iter().enumerate() {
                let expected = if j == m { 1.0 } else { 0.0 };
                assert!((horner(coeffs, t_m) - expected).abs() < 1e-10);
            }
        }
        // Σ_j L_j(α) = 1 for any α.
        for &alpha in &[-1.3, 0.7, 2.2, 5.0] {
            let total: f64 = basis.iter().map(|c| horner(c, alpha)).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "partition of unity failed at {alpha}"
            );
        }
    }

    #[test]
    fn repeated_lagrange_nodes_are_rejected() {
        assert!(lagrange_monomial_coeffs(&[0.0, 1.0, 1.0]).is_err());
    }
}
