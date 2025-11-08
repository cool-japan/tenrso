//! Representation selection (dense/sparse/low-rank)
//!
//! Provides heuristics for selecting optimal tensor representations based on
//! sparsity, size, and computational cost.

use crate::api::ReprHint;
use crate::cost::TensorStats;

/// Configuration for representation selection
#[derive(Debug, Clone)]
pub struct ReprConfig {
    /// Sparsity threshold above which to use sparse representation (default: 0.9)
    pub sparse_threshold: f64,

    /// Minimum tensor size (number of elements) to consider sparse (default: 10,000)
    /// Small tensors stay dense even if sparse, due to overhead
    pub min_sparse_size: usize,

    /// Low-rank threshold: max ratio of rank to min dimension (default: 0.3)
    /// If estimated_rank / min(dims) < threshold, consider low-rank
    pub lowrank_threshold: f64,

    /// Memory pressure factor (0.0 = no pressure, 1.0 = extreme pressure)
    /// Higher pressure favors more compressed representations
    pub memory_pressure: f64,
}

impl Default for ReprConfig {
    fn default() -> Self {
        Self {
            sparse_threshold: 0.9,
            min_sparse_size: 10_000,
            lowrank_threshold: 0.3,
            memory_pressure: 0.0,
        }
    }
}

/// Select optimal representation for a tensor based on statistics and configuration
///
/// # Arguments
///
/// * `stats` - Tensor statistics (shape, nnz, density)
/// * `config` - Configuration for representation selection
///
/// # Returns
///
/// Recommended representation hint
///
/// # Algorithm
///
/// 1. If tensor is small (< min_sparse_size), prefer dense
/// 2. If sparsity > sparse_threshold and size > min_sparse_size, use sparse
/// 3. If low-rank characteristics detected, consider low-rank
/// 4. Otherwise, use dense
pub fn select_representation(stats: &TensorStats, config: &ReprConfig) -> ReprHint {
    let size = stats.size();
    let sparsity = 1.0 - stats.density;

    // Rule 1: Small tensors should stay dense (overhead not worth it)
    if size < config.min_sparse_size {
        return ReprHint::Dense;
    }

    // Rule 2: High sparsity → sparse representation
    if sparsity >= config.sparse_threshold {
        return ReprHint::Sparse;
    }

    // Rule 3: Check for low-rank characteristics
    // For now, we use a simple heuristic: if one dimension is much smaller than others,
    // it might benefit from low-rank representation
    if is_lowrank_candidate(stats, config) {
        return ReprHint::LowRank;
    }

    // Rule 4: Medium sparsity with memory pressure → sparse
    if sparsity > 0.5 && config.memory_pressure > 0.5 {
        return ReprHint::Sparse;
    }

    // Default: Dense
    ReprHint::Dense
}

/// Check if tensor is a good candidate for low-rank representation
fn is_lowrank_candidate(stats: &TensorStats, config: &ReprConfig) -> bool {
    if stats.shape.len() < 2 {
        return false; // Need at least 2D for low-rank
    }

    // Find minimum dimension
    let min_dim = stats.shape.iter().copied().min().unwrap_or(0);
    let max_dim = stats.shape.iter().copied().max().unwrap_or(0);

    if min_dim == 0 || max_dim == 0 {
        return false;
    }

    // If min dimension is small relative to max, could benefit from low-rank
    // (e.g., 1000×10 matrix could be rank-10 at most)
    let dimension_ratio = min_dim as f64 / max_dim as f64;

    dimension_ratio < config.lowrank_threshold
}

/// Estimate memory savings from using sparse representation
///
/// Returns ratio of sparse_memory / dense_memory
pub fn sparse_memory_ratio(stats: &TensorStats) -> f64 {
    if stats.nnz.is_none() {
        return 1.0; // Dense tensor
    }

    let nnz = stats.nnz.unwrap();
    let total = stats.size();

    if total == 0 {
        return 1.0;
    }

    // Sparse format overhead:
    // - COO: 3 arrays (row, col, val) for 2D, more for N-D
    // - CSR: row_ptr (n+1) + col_idx (nnz) + val (nnz)
    // - Estimate: ~2.5× nnz for 2D sparse formats

    let sparse_memory = if stats.shape.len() == 2 {
        // CSR/CSC: row_ptr + col_idx + values
        (stats.shape[0] + 1 + nnz * 2) as f64
    } else {
        // COO for N-D: N coordinate arrays + values
        (nnz * (stats.shape.len() + 1)) as f64
    };

    let dense_memory = total as f64;

    sparse_memory / dense_memory
}

/// Estimate memory savings from using low-rank representation
///
/// Returns ratio of lowrank_memory / dense_memory
///
/// # Arguments
///
/// * `stats` - Tensor statistics
/// * `rank` - Target rank for low-rank approximation
pub fn lowrank_memory_ratio(stats: &TensorStats, rank: usize) -> f64 {
    if stats.shape.len() < 2 {
        return 1.0; // Not applicable
    }

    let total = stats.size();
    if total == 0 {
        return 1.0;
    }

    // For CP decomposition: sum of factor matrix sizes
    // For N-mode tensor with rank R: sum_n (dim_n * R)
    let cp_memory: usize = stats.shape.iter().map(|&dim| dim * rank).sum();

    let dense_memory = total;

    cp_memory as f64 / dense_memory as f64
}

/// Select representation for a contraction result
///
/// # Arguments
///
/// * `input_stats` - Statistics for input tensors
/// * `output_stats` - Statistics for output tensor
/// * `config` - Configuration for representation selection
///
/// # Returns
///
/// Recommended representation for the output
pub fn select_output_representation(
    input_stats: &[TensorStats],
    output_stats: &TensorStats,
    config: &ReprConfig,
) -> ReprHint {
    // If all inputs are sparse, output likely sparse
    let all_inputs_sparse = input_stats.iter().all(|s| s.is_sparse());

    if all_inputs_sparse {
        // Check if output would benefit from sparse
        let output_sparsity = 1.0 - output_stats.density;
        if output_sparsity >= config.sparse_threshold {
            return ReprHint::Sparse;
        }
    }

    // Otherwise, use standard selection
    select_representation(output_stats, config)
}

/// Compute the cost of converting between representations
///
/// # Arguments
///
/// * `from` - Source representation
/// * `to` - Target representation
/// * `stats` - Tensor statistics
///
/// # Returns
///
/// Estimated cost in FLOPs for the conversion
pub fn conversion_cost(from: ReprHint, to: ReprHint, stats: &TensorStats) -> f64 {
    use ReprHint::*;

    match (from, to) {
        (Dense, Dense) | (Sparse, Sparse) | (LowRank, LowRank) => 0.0, // No conversion

        (Dense, Sparse) | (Sparse, Dense) => {
            // Conversion cost: iterate through all elements
            stats.size() as f64
        }

        (Dense, LowRank) | (Sparse, LowRank) => {
            // Low-rank decomposition cost (expensive!)
            // Roughly O(n * r^2) for CP-ALS where n is size, r is rank
            let n = stats.size() as f64;
            let r = 64.0; // Assume rank 64
            n * r * r
        }

        (LowRank, Dense) => {
            // Reconstruction cost: sum of outer products
            // O(n * r) where n is size, r is rank
            let n = stats.size() as f64;
            let r = 64.0;
            n * r
        }

        (LowRank, Sparse) => {
            // Reconstruct then convert to sparse
            let recon_cost = conversion_cost(LowRank, Dense, stats);
            let sparse_cost = conversion_cost(Dense, Sparse, stats);
            recon_cost + sparse_cost
        }

        (Auto, _) | (_, Auto) => {
            // Auto is a hint, not a real representation
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_representation_dense_small() {
        let stats = TensorStats::dense(vec![10, 10]);
        let config = ReprConfig::default();

        let repr = select_representation(&stats, &config);
        assert_eq!(repr, ReprHint::Dense);
    }

    #[test]
    fn test_select_representation_sparse() {
        // Large sparse tensor (95% sparse)
        let stats = TensorStats::sparse(vec![1000, 1000], 50_000);
        let config = ReprConfig::default();

        let repr = select_representation(&stats, &config);
        assert_eq!(repr, ReprHint::Sparse);
    }

    #[test]
    fn test_select_representation_dense_medium_sparsity() {
        // Medium sparsity (60%), should stay dense
        let stats = TensorStats::sparse(vec![1000, 1000], 400_000);
        let config = ReprConfig::default();

        let repr = select_representation(&stats, &config);
        assert_eq!(repr, ReprHint::Dense);
    }

    #[test]
    fn test_select_representation_lowrank() {
        // Very rectangular matrix: 1000×10 (could be at most rank 10)
        let stats = TensorStats::dense(vec![1000, 10]);
        let config = ReprConfig {
            lowrank_threshold: 0.05,
            ..Default::default()
        };

        let repr = select_representation(&stats, &config);
        assert_eq!(repr, ReprHint::LowRank);
    }

    #[test]
    fn test_select_representation_memory_pressure() {
        // 60% sparse, normally dense, but high memory pressure
        let stats = TensorStats::sparse(vec![1000, 1000], 400_000);
        let config = ReprConfig {
            memory_pressure: 0.8,
            ..Default::default()
        };

        let repr = select_representation(&stats, &config);
        assert_eq!(repr, ReprHint::Sparse);
    }

    #[test]
    fn test_is_lowrank_candidate() {
        let config = ReprConfig::default();

        // Rectangular matrix
        let stats = TensorStats::dense(vec![1000, 100]);
        assert!(is_lowrank_candidate(&stats, &config));

        // Square matrix
        let stats = TensorStats::dense(vec![1000, 1000]);
        assert!(!is_lowrank_candidate(&stats, &config));

        // 1D tensor
        let stats = TensorStats::dense(vec![1000]);
        assert!(!is_lowrank_candidate(&stats, &config));
    }

    #[test]
    fn test_sparse_memory_ratio() {
        // 5% dense (95% sparse)
        let stats = TensorStats::sparse(vec![1000, 1000], 50_000);
        let ratio = sparse_memory_ratio(&stats);

        // Sparse should be much smaller than dense
        assert!(ratio < 0.2);
    }

    #[test]
    fn test_sparse_memory_ratio_dense() {
        let stats = TensorStats::dense(vec![100, 100]);
        let ratio = sparse_memory_ratio(&stats);

        // Dense tensor has ratio 1.0
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_lowrank_memory_ratio() {
        // 1000×1000 matrix with rank 50
        let stats = TensorStats::dense(vec![1000, 1000]);
        let ratio = lowrank_memory_ratio(&stats, 50);

        // Low-rank: (1000 + 1000) * 50 = 100,000
        // Dense: 1,000,000
        // Ratio: 0.1
        assert!((ratio - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_lowrank_memory_ratio_3d() {
        // 100×100×100 tensor with rank 10
        let stats = TensorStats::dense(vec![100, 100, 100]);
        let ratio = lowrank_memory_ratio(&stats, 10);

        // Low-rank: (100 + 100 + 100) * 10 = 3,000
        // Dense: 1,000,000
        // Ratio: 0.003
        assert!((ratio - 0.003).abs() < 0.001);
    }

    #[test]
    fn test_select_output_representation_all_sparse() {
        let input_stats = vec![
            TensorStats::sparse(vec![1000, 1000], 10_000),
            TensorStats::sparse(vec![1000, 1000], 10_000),
        ];
        let output_stats = TensorStats::sparse(vec![1000, 1000], 5_000);
        let config = ReprConfig::default();

        let repr = select_output_representation(&input_stats, &output_stats, &config);
        assert_eq!(repr, ReprHint::Sparse);
    }

    #[test]
    fn test_select_output_representation_mixed() {
        let input_stats = vec![
            TensorStats::dense(vec![100, 100]),
            TensorStats::sparse(vec![100, 100], 1_000),
        ];
        let output_stats = TensorStats::dense(vec![100, 100]);
        let config = ReprConfig::default();

        let repr = select_output_representation(&input_stats, &output_stats, &config);
        assert_eq!(repr, ReprHint::Dense);
    }

    #[test]
    fn test_conversion_cost_same() {
        let stats = TensorStats::dense(vec![100, 100]);

        let cost = conversion_cost(ReprHint::Dense, ReprHint::Dense, &stats);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_conversion_cost_dense_sparse() {
        let stats = TensorStats::dense(vec![100, 100]);

        let cost = conversion_cost(ReprHint::Dense, ReprHint::Sparse, &stats);
        assert_eq!(cost, 10_000.0); // size of tensor
    }

    #[test]
    fn test_conversion_cost_dense_lowrank() {
        let stats = TensorStats::dense(vec![100, 100]);

        let cost = conversion_cost(ReprHint::Dense, ReprHint::LowRank, &stats);
        // Should be expensive (decomposition cost)
        assert!(cost > 100_000.0);
    }

    #[test]
    fn test_conversion_cost_lowrank_dense() {
        let stats = TensorStats::dense(vec![100, 100]);

        let cost = conversion_cost(ReprHint::LowRank, ReprHint::Dense, &stats);
        // Reconstruction cost: n * r
        assert!((cost - 640_000.0).abs() < 1.0); // 10,000 * 64
    }

    #[test]
    fn test_config_defaults() {
        let config = ReprConfig::default();

        assert_eq!(config.sparse_threshold, 0.9);
        assert_eq!(config.min_sparse_size, 10_000);
        assert_eq!(config.lowrank_threshold, 0.3);
        assert_eq!(config.memory_pressure, 0.0);
    }
}
