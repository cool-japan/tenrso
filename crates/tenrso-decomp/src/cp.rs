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
}

impl<T> CpDecomp<T>
where
    T: Float + FloatConst + NumCast,
{
    /// Reconstruct the original tensor from the CP decomposition
    ///
    /// Computes X ≈ Σᵣ λᵣ (u₁ᵣ ⊗ u₂ᵣ ⊗ ... ⊗ uₙᵣ)
    ///
    /// # Complexity
    ///
    /// Time: O(R × ∏ᵢ Iᵢ)
    /// Space: O(∏ᵢ Iᵢ)
    pub fn reconstruct(&self, shape: &[usize]) -> Result<DenseND<T>> {
        let rank = self.factors[0].shape()[1];
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

        // Initialize reconstruction tensor
        let total_size: usize = shape.iter().product();
        let mut data = vec![T::zero(); total_size];

        // For each rank-1 component
        for r in 0..rank {
            let weight = self.weights.as_ref().map_or(T::one(), |w| w[r]);

            // Compute outer product of all factor vectors
            // This is expensive but straightforward
            #[allow(clippy::needless_range_loop)]
            for idx in 0..total_size {
                let mut value = weight;
                let mut remaining = idx;

                // Convert linear index to multi-index
                for mode in (0..n_modes).rev() {
                    let mode_size = shape[mode];
                    let mode_idx = remaining % mode_size;
                    remaining /= mode_size;

                    value = value * self.factors[mode][[mode_idx, r]];
                }

                data[idx] = data[idx] + value;
            }
        }

        DenseND::from_vec(data, shape)
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
/// // Decompose with rank 5
/// let cp = cp_als(&tensor, 5, 50, 1e-4, InitStrategy::Random).unwrap();
///
/// println!("Final fit: {:.4}", cp.fit);
/// println!("Iterations: {}", cp.iters);
/// ```
pub fn cp_als<T>(
    tensor: &DenseND<T>,
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

    // ALS iterations
    for iter in 0..max_iters {
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

        // Check convergence
        let fit_change = (fit - prev_fit).abs();
        if iter > 0 && fit_change < NumCast::from(tol).unwrap() {
            break;
        }

        prev_fit = fit;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
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
/// let cp = cp_als_constrained(&tensor, 5, 50, 1e-4, InitStrategy::Random, constraints).unwrap();
/// ```
pub fn cp_als_constrained<T>(
    tensor: &DenseND<T>,
    rank: usize,
    max_iters: usize,
    tol: f64,
    init: InitStrategy,
    constraints: CpConstraints,
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

    // ALS iterations
    for iter in 0..max_iters {
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

        // Check convergence
        let fit_change = (fit - prev_fit).abs();
        if iter > 0 && fit_change < NumCast::from(tol).unwrap() {
            break;
        }

        prev_fit = fit;
    }

    Ok(CpDecomp {
        factors,
        weights: None,
        fit,
        iters,
    })
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
    }

    Ok(factors)
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
        // Small tensor for quick test
        let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
        let result = cp_als(&tensor, 2, 10, 1e-4, InitStrategy::Random);

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
        let result = cp_als_constrained(&tensor, 3, 20, 1e-4, InitStrategy::Random, constraints);

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
        let result = cp_als_constrained(&tensor, 3, 20, 1e-4, InitStrategy::Random, constraints);

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
        let result = cp_als_constrained(&tensor, 4, 10, 1e-4, InitStrategy::Random, constraints);

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
        let result = cp_als_constrained(&tensor, 3, 20, 1e-4, InitStrategy::Random, constraints);

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
}
