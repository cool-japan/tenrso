//! Utility functions for decomposition analysis and comparison
//!
//! This module provides tools for:
//! - Comparing different decomposition methods
//! - Analyzing decomposition quality
//! - Estimating appropriate ranks
//! - Computing factor statistics

use anyhow::Result;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::numeric::{Float, NumCast};
use std::iter::Sum;
use tenrso_core::DenseND;

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
    let normalizer = T::from(rank).unwrap().sqrt();
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
    let threshold = s_max * T::from(sv_threshold).unwrap();
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

    let min_dim = *shape.iter().min().unwrap();
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

    #[allow(clippy::needless_range_loop)] // mode is needed for unfold(mode), not just indexing
    for mode in 0..n_modes {
        // Unfold tensor along this mode
        let unfolded = tensor.unfold(mode)?;

        // Compute SVD
        let (_, s, _) = svd(&unfolded.view(), false, None)?;

        // Find rank that preserves energy_threshold of energy
        let total_energy: T = s.iter().map(|&sigma| sigma * sigma).sum();
        let target_energy = total_energy * T::from(energy_threshold).unwrap();

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
        use scirs2_core::random::{thread_rng, Rng};

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
