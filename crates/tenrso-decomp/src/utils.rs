//! Utility functions for decomposition analysis and comparison
//!
//! This module provides tools for:
//! - Comparing different decomposition methods
//! - Analyzing decomposition quality
//! - Estimating appropriate ranks
//! - Computing factor statistics

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, NumCast, ToPrimitive};
use std::iter::Sum;
use tenrso_core::DenseND;

/// Cast a primitive literal to type T.
///
/// Panics if the cast is impossible (e.g., a usize that doesn't fit in f32).
/// Only intended for small compile-time constants — not for user data.
#[inline]
pub(crate) fn cast_lit<T, L>(val: L) -> T
where
    T: NumCast,
    L: ToPrimitive,
{
    NumCast::from(val).expect("cast_lit: numeric cast failed for constant")
}

/// Statistics for a decomposition method
#[derive(Debug, Clone)]
pub struct DecompStats<T> {
    /// Relative reconstruction error: ||X - X_approx|| / ||X||
    pub relative_error: T,

    /// Compression ratio: original_size / compressed_size
    pub compression_ratio: f64,

    /// Number of parameters in the decomposition
    pub num_parameters: usize,

    /// Original tensor size
    pub original_size: usize,

    /// Method name (e.g., "CP", "Tucker", "TT")
    pub method: String,
}

impl<T: Float> DecompStats<T> {
    /// Compute quality score combining error and compression
    ///
    /// Score = compression_ratio / (1 + error)
    /// Higher is better (more compression with less error)
    pub fn quality_score(&self) -> f64 {
        let error_f64: f64 = NumCast::from(self.relative_error).unwrap_or(1.0);
        self.compression_ratio / (1.0 + error_f64)
    }

    /// Check if decomposition meets quality threshold
    pub fn meets_threshold(&self, max_error: T) -> bool {
        self.relative_error <= max_error
    }
}

/// Factor matrix statistics for quality analysis
#[derive(Debug, Clone)]
pub struct FactorStats<T> {
    /// Orthogonality measure: ||U^T U - I||_F / sqrt(rank)
    /// 0 = perfectly orthogonal, larger values indicate less orthogonality
    pub orthogonality_error: T,

    /// Condition number of the factor matrix
    pub condition_number: T,

    /// Frobenius norm of the factor
    pub frobenius_norm: T,

    /// Number of effective components (based on singular value decay)
    pub effective_rank: usize,
}

/// Compute orthogonality error for a factor matrix
///
/// Measures how close U^T U is to the identity matrix.
/// Returns ||U^T U - I||_F / sqrt(rank)
///
/// # Arguments
///
/// * `factor` - Factor matrix with shape (n, rank)
///
/// # Returns
///
/// Normalized orthogonality error (0 = perfectly orthogonal)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array2;
/// use tenrso_decomp::utils::compute_orthogonality_error;
///
/// // Create an orthogonal matrix (columns of identity)
/// let mut factor = Array2::<f64>::zeros((5, 3));
/// for i in 0..3 {
///     factor[[i, i]] = 1.0;
/// }
///
/// let error = compute_orthogonality_error(&factor);
/// assert!(error < 1e-10, "Orthogonality error should be near zero");
/// ```
pub fn compute_orthogonality_error<T>(factor: &Array2<T>) -> T
where
    T: Float + NumCast + Sum + scirs2_core::ndarray_ext::ScalarOperand + 'static,
{
    let rank = factor.ncols();

    // Compute U^T U
    let gram = factor.t().dot(factor);

    // Compute ||U^T U - I||_F²
    let mut error_sq = T::zero();
    for i in 0..rank {
        for j in 0..rank {
            let expected = if i == j { T::one() } else { T::zero() };
            let diff = gram[[i, j]] - expected;
            error_sq = error_sq + diff * diff;
        }
    }

    // Normalize by sqrt(rank)
    let normalizer: T = cast_lit::<T, _>(rank).sqrt();
    (error_sq.sqrt()) / normalizer
}

/// Analyze factor matrix quality
///
/// Computes statistics including orthogonality, condition number, and effective rank.
///
/// # Arguments
///
/// * `factor` - Factor matrix to analyze
/// * `sv_threshold` - Threshold for effective rank (default: 0.01)
///
/// # Returns
///
/// FactorStats containing quality metrics
pub fn analyze_factor<T>(factor: &Array2<T>, sv_threshold: f64) -> Result<FactorStats<T>>
where
    T: Float
        + NumCast
        + Sum
        + scirs2_core::numeric::NumAssign
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    use scirs2_linalg::svd;

    // Compute orthogonality error
    let orthogonality_error = compute_orthogonality_error(factor);

    // Compute SVD for condition number and effective rank
    let (_, s, _) = svd(&factor.view(), false, None)?;

    let s_max = s[0];
    let s_min = s[s.len() - 1];
    let condition_number = s_max / s_min;

    // Effective rank: count singular values > threshold * max
    let threshold_t: T = NumCast::from(sv_threshold).ok_or_else(|| {
        anyhow::anyhow!("could not convert sv_threshold={sv_threshold} to scalar type")
    })?;
    let threshold = s_max * threshold_t;
    let effective_rank = s.iter().filter(|&&sigma| sigma > threshold).count();

    // Frobenius norm
    let mut norm_sq = T::zero();
    for &val in factor.iter() {
        norm_sq += val * val;
    }
    let frobenius_norm = norm_sq.sqrt();

    Ok(FactorStats {
        orthogonality_error,
        condition_number,
        frobenius_norm,
        effective_rank,
    })
}

/// Compare reconstruction errors across multiple decompositions
///
/// Helper function to evaluate which decomposition method performs best
/// for a given tensor and rank budget.
///
/// # Arguments
///
/// * `original` - Original tensor
/// * `reconstructions` - List of (method_name, reconstructed_tensor) pairs
///
/// # Returns
///
/// Vector of (method_name, relative_error) sorted by error (best first)
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::{cp_als, tucker_hosvd, InitStrategy};
/// use tenrso_decomp::utils::compare_reconstructions;
///
/// let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
///
/// // Compute different decompositions
/// let cp = cp_als(&tensor, 5, 50, 1e-4, InitStrategy::Random, None).unwrap();
/// let tucker = tucker_hosvd(&tensor, &[5, 5, 5]).unwrap();
///
/// let cp_recon = cp.reconstruct(tensor.shape()).unwrap();
/// let tucker_recon = tucker.reconstruct().unwrap();
///
/// // Compare methods
/// let comparisons = compare_reconstructions(
///     &tensor,
///     vec![("CP-5", cp_recon), ("Tucker-5", tucker_recon)]
/// ).unwrap();
///
/// for (method, error) in comparisons {
///     println!("{}: {:.6}", method, error);
/// }
/// ```
pub fn compare_reconstructions<T>(
    original: &DenseND<T>,
    reconstructions: Vec<(&str, DenseND<T>)>,
) -> Result<Vec<(String, T)>>
where
    T: Float + NumCast + PartialOrd + Sum + scirs2_core::numeric::FromPrimitive,
{
    let orig_norm = original.frobenius_norm();

    let mut results = Vec::new();

    for (method, recon) in reconstructions {
        // Compute error
        let mut error_sq = T::zero();
        for (orig_val, recon_val) in original.view().iter().zip(recon.view().iter()) {
            let diff = *orig_val - *recon_val;
            error_sq = error_sq + diff * diff;
        }

        let relative_error = error_sq.sqrt() / orig_norm;
        results.push((method.to_string(), relative_error));
    }

    // Sort by error (best first)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Estimate CP rank using heuristic based on tensor dimensions
///
/// Provides a starting point for CP decomposition rank selection.
/// Uses heuristics based on tensor size and available memory.
///
/// # Arguments
///
/// * `shape` - Tensor shape
/// * `max_rank_ratio` - Maximum rank as fraction of smallest dimension (default: 0.5)
///
/// # Returns
///
/// Suggested CP rank
///
/// # Examples
///
/// ```
/// use tenrso_decomp::utils::estimate_cp_rank;
///
/// let shape = vec![100, 100, 100];
/// let suggested_rank = estimate_cp_rank(&shape, 0.5);
///
/// println!("Suggested CP rank: {}", suggested_rank);
/// assert!(suggested_rank > 0);
/// assert!(suggested_rank <= 50); // At most 50% of smallest dimension
/// ```
pub fn estimate_cp_rank(shape: &[usize], max_rank_ratio: f64) -> usize {
    if shape.is_empty() {
        return 1;
    }

    // `shape.is_empty()` has been checked above, so `min()` is guaranteed
    // to return `Some`. We defensively fall back to `1` if a future refactor
    // bypasses the emptiness guard, which preserves the semantic "no rank
    // smaller than 1" without panicking.
    let min_dim = shape.iter().min().copied().unwrap_or(1);
    let n_modes = shape.len();

    // Heuristic: rank ~ min_dim^(1/n_modes)
    // But capped at max_rank_ratio * min_dim
    let suggested = (min_dim as f64).powf(1.0 / n_modes as f64).ceil() as usize;
    let max_rank = (min_dim as f64 * max_rank_ratio).floor() as usize;

    suggested.min(max_rank).max(1)
}

/// Estimate Tucker ranks using energy-based heuristic
///
/// Suggests Tucker ranks that would preserve a target percentage of
/// the tensor's "energy" (sum of squared singular values).
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `energy_threshold` - Target energy preservation (e.g., 0.9 for 90%)
///
/// # Returns
///
/// Vector of suggested ranks for each mode
///
/// # Examples
///
/// ```
/// use tenrso_core::DenseND;
/// use tenrso_decomp::utils::estimate_tucker_ranks;
///
/// let tensor = DenseND::<f64>::random_uniform(&[20, 20, 20], 0.0, 1.0);
/// let suggested_ranks = estimate_tucker_ranks(&tensor, 0.9).unwrap();
///
/// println!("Suggested Tucker ranks: {:?}", suggested_ranks);
/// ```
pub fn estimate_tucker_ranks<T>(tensor: &DenseND<T>, energy_threshold: f64) -> Result<Vec<usize>>
where
    T: Float
        + NumCast
        + scirs2_core::numeric::NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    use scirs2_linalg::svd;

    let shape = tensor.shape();
    let n_modes = tensor.rank();
    let mut ranks = Vec::with_capacity(n_modes);
    let energy_threshold_t: T = NumCast::from(energy_threshold).ok_or_else(|| {
        anyhow::anyhow!("could not convert energy_threshold={energy_threshold} to scalar type")
    })?;

    #[allow(clippy::needless_range_loop)] // mode is needed for unfold(mode), not just indexing
    for mode in 0..n_modes {
        // Unfold tensor along this mode
        let unfolded = tensor.unfold(mode)?;

        // Compute SVD
        let (_, s, _) = svd(&unfolded.view(), false, None)?;

        // Find rank that preserves energy_threshold of energy
        let total_energy: T = s.iter().map(|&sigma| sigma * sigma).sum();
        let target_energy = total_energy * energy_threshold_t;

        let mut cumulative_energy = T::zero();
        let mut rank = 1;

        for (i, &sigma) in s.iter().enumerate() {
            cumulative_energy += sigma * sigma;
            rank = i + 1;

            if cumulative_energy >= target_energy {
                break;
            }
        }

        ranks.push(rank.min(shape[mode]));
    }

    Ok(ranks)
}

/// Compute a truncated SVD of a short-fat matrix (rows << cols) via the Gram matrix.
///
/// For an `m × n` matrix A where m << n, full SVD costs O(m² n) — the same as forming
/// G = A Aᵀ (m×m) and eigendecomposing it. The advantage over randomized SVD is that
/// Gram SVD avoids allocating an n×(k+oversampling) Gaussian Omega matrix, which would
/// be O(nk) ≈ O(nm) bytes — potentially gigabytes for very wide TT unfoldings.
///
/// # Numerical note
/// Forming G squares the condition number. For TT-SVD truncation at moderate tolerances
/// (1e-6 to 1e-10) this is acceptable. Do not use for high-precision singular-vector needs.
///
/// # Arguments
/// * `matrix` — m × n view, must have m ≤ n
/// * `target_rank` — number of singular triplets to return (clamped to m)
///
/// # Returns
/// (U, S, Vt) where U is m×k, S is k, Vt is k×n
pub(crate) fn thin_svd_via_gram<T>(
    matrix: &scirs2_core::ndarray_ext::ArrayView2<T>,
    target_rank: usize,
) -> anyhow::Result<(Array2<T>, Array1<T>, Array2<T>)>
where
    T: Float + NumCast + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_linalg::svd;

    let (m, _n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = target_rank.min(m);

    // Step 1: Gram matrix G = A Aᵀ  (m×m),  cost O(m² n)
    // For short-fat A, m is tiny so G is cheap to eigendecompose.
    let g = matrix.dot(&matrix.t());

    // Step 2: SVD of G.  G is symmetric PSD, so its SVD = eigendecomposition:
    //   G = U diag(λ) Uᵀ,  λᵢ = σᵢ²(A)
    let (u_full, lambda, _vt_g) = svd(&g.view(), false, None)?;

    // Step 3: Truncate U and derive singular values σ = sqrt(λ)
    let u_k = u_full.slice(s![.., ..k]).to_owned();
    let sigma_k: Array1<T> = lambda.slice(s![..k]).mapv(|lam| lam.max(T::zero()).sqrt());

    // Step 4: Recover Vᵀ = diag(1/σ) · Uᵀ · A   (k×n),  cost O(k m n)
    // Build the scaled Uᵀ first (k×m, cheap), then multiply by A.
    let eps = T::epsilon() * cast_lit::<T, _>(1000_u64);
    let mut ut_scaled = u_k.t().to_owned(); // k×m
    for i in 0..k {
        let inv_sigma = if sigma_k[i] > eps {
            T::one() / sigma_k[i]
        } else {
            T::zero()
        };
        for j in 0..m {
            ut_scaled[[i, j]] *= inv_sigma;
        }
    }
    let vt_k = ut_scaled.dot(matrix); // (k×m) · (m×n) = k×n

    Ok((u_k, sigma_k, vt_k))
}

/// Decide whether to use randomized SVD for an m×n matrix with target rank k.
///
/// Randomized SVD is profitable when the target rank is at most half the smaller
/// dimension (so the sketch is genuinely low-rank) and the matrix is large enough
/// that the constant-factor overhead is worthwhile.
///
/// # Arguments
/// * `rows` — number of matrix rows
/// * `cols` — number of matrix columns
/// * `target_rank` — desired number of singular triplets
pub(crate) fn should_use_randomized_svd(rows: usize, cols: usize, target_rank: usize) -> bool {
    let min_dim = rows.min(cols);
    target_rank <= min_dim / 2 && min_dim > 32
}

/// Compute truncated SVD using randomized algorithm (Halko-Martinsson-Tropp 2011).
///
/// For an m x n matrix, computes an approximate rank-k SVD:
///   A ≈ U_k Σ_k V_k^T
///
/// This is dramatically faster than full SVD when k << min(m,n).
/// Complexity: O(m * n * k) instead of O(m * n * min(m,n)).
///
/// # Arguments
///
/// * `matrix` - Input m x n matrix (as ArrayView2)
/// * `target_rank` - Number of singular values/vectors to compute
/// * `oversampling` - Extra samples for accuracy (typically 5-10)
/// * `power_iters` - Power iterations for better approximation (typically 1-2)
///
/// # Returns
///
/// (U, S, Vt) where U is m x k, S is k, Vt is k x n
pub fn randomized_svd_truncated<T>(
    matrix: &scirs2_core::ndarray_ext::ArrayView2<T>,
    target_rank: usize,
    oversampling: usize,
    power_iters: usize,
) -> Result<(Array2<T>, scirs2_core::ndarray_ext::Array1<T>, Array2<T>)>
where
    T: Float
        + NumCast
        + scirs2_core::numeric::NumAssign
        + Sum
        + scirs2_core::ndarray_ext::ScalarOperand
        + Send
        + Sync
        + std::fmt::Debug
        + 'static,
{
    use scirs2_core::random::{thread_rng, Distribution, RandNormal as Normal};
    use scirs2_linalg::{qr, svd};

    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = target_rank.min(m).min(n);
    let l = (k + oversampling).min(m).min(n);

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| anyhow::anyhow!("failed to construct Normal(0.0, 1.0): {e}"))?;

    // Step 1: Generate random Gaussian matrix Omega (n x l)
    let omega = Array2::<T>::from_shape_fn((n, l), |_| cast_lit(normal.sample(&mut rng)));

    // Step 2: Form the sample matrix Y = A * Omega (m x l)
    let mut y = matrix.dot(&omega);

    // Step 3: Power iterations for improved accuracy
    for _ in 0..power_iters {
        // y = A * (A^T * y)
        let aty = matrix.t().dot(&y.view());
        y = matrix.dot(&aty.view());
    }

    // Step 4: QR decomposition of Y = Q * R
    let (q, _r) = qr(&y.view(), None)?;

    // Take only l columns from Q (thin QR)
    let q_thin = if q.shape()[1] > l {
        q.slice(scirs2_core::ndarray_ext::s![.., ..l]).to_owned()
    } else {
        q
    };

    // Step 5: Form B = Q^T * A (l x n) — small matrix
    let b = q_thin.t().dot(matrix);

    // Step 6: SVD of small matrix B = U_b * S * Vt
    let (u_b, s, vt) = svd(&b.view(), false, None)?;

    // Step 7: Recover left singular vectors: U = Q * U_b
    let u_full = q_thin.dot(&u_b.view());

    // Truncate to target rank k
    let u_trunc = u_full
        .slice(scirs2_core::ndarray_ext::s![.., ..k])
        .to_owned();
    let s_trunc = s.slice(scirs2_core::ndarray_ext::s![..k]).to_owned();
    let vt_trunc = vt.slice(scirs2_core::ndarray_ext::s![..k, ..]).to_owned();

    Ok((u_trunc, s_trunc, vt_trunc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;

    #[test]
    fn test_orthogonality_error_identity() {
        // Create orthogonal matrix (part of identity)
        let mut factor = Array2::<f64>::zeros((5, 3));
        for i in 0..3 {
            factor[[i, i]] = 1.0;
        }

        let error = compute_orthogonality_error(&factor);
        assert!(
            error < 1e-10,
            "Orthogonal matrix should have near-zero error"
        );
    }

    #[test]
    fn test_orthogonality_error_random() {
        use scirs2_core::random::thread_rng;

        // Random matrix should have non-zero orthogonality error
        let mut rng = thread_rng();
        let factor = Array2::<f64>::from_shape_fn((10, 5), |_| rng.random::<f64>());

        let error = compute_orthogonality_error(&factor);
        assert!(
            error > 0.01,
            "Random matrix should have significant orthogonality error"
        );
    }

    #[test]
    fn test_analyze_factor() {
        // Create a well-conditioned factor
        let mut factor = Array2::<f64>::zeros((10, 5));
        for i in 0..5 {
            factor[[i, i]] = 1.0;
        }

        let stats = analyze_factor(&factor, 0.01).unwrap();

        assert!(stats.orthogonality_error < 0.1);
        assert!(stats.effective_rank >= 1);
        assert!(stats.frobenius_norm > 0.0);
    }

    #[test]
    fn test_compare_reconstructions() {
        let original = DenseND::<f64>::ones(&[5, 5, 5]);

        // Perfect reconstruction
        let recon1 = DenseND::<f64>::ones(&[5, 5, 5]);

        // Imperfect reconstruction
        let recon2 = DenseND::<f64>::from_vec(vec![0.9; 125], &[5, 5, 5]).unwrap();

        let results =
            compare_reconstructions(&original, vec![("Perfect", recon1), ("Imperfect", recon2)])
                .unwrap();

        // Perfect should be first (lower error)
        assert_eq!(results[0].0, "Perfect");
        assert!(results[0].1 < results[1].1);
    }

    #[test]
    fn test_estimate_cp_rank() {
        let shape = vec![100, 100, 100];
        let rank = estimate_cp_rank(&shape, 0.5);

        assert!(rank > 0);
        assert!(rank <= 50); // Max 50% of min dimension
    }

    #[test]
    fn test_estimate_tucker_ranks() {
        let tensor = DenseND::<f64>::random_uniform(&[10, 10, 10], 0.0, 1.0);
        let ranks = estimate_tucker_ranks(&tensor, 0.9).unwrap();

        assert_eq!(ranks.len(), 3);
        for &rank in &ranks {
            assert!(rank > 0);
            assert!(rank <= 10);
        }
    }

    #[test]
    fn test_decomp_stats_quality_score() {
        let stats = DecompStats {
            relative_error: 0.1,
            compression_ratio: 10.0,
            num_parameters: 1000,
            original_size: 10000,
            method: "Test".to_string(),
        };

        let score = stats.quality_score();
        assert!(score > 0.0);

        // Higher compression and lower error should give better score
        let better_stats = DecompStats {
            relative_error: 0.05,
            compression_ratio: 20.0,
            ..stats.clone()
        };

        assert!(better_stats.quality_score() > stats.quality_score());
    }
}
