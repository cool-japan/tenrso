//! Cost models for tensor contractions
//!
//! Provides functions to estimate computational cost (FLOPs) and memory usage
//! for tensor contractions. Supports both dense and sparse tensors.

use crate::parser::EinsumSpec;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Statistics about a tensor for cost estimation
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Number of non-zero elements (for sparse tensors)
    /// If None, tensor is assumed to be dense
    pub nnz: Option<usize>,
    /// Density (0.0 to 1.0), computed from nnz if available
    pub density: f64,
}

impl TensorStats {
    /// Create stats for a dense tensor
    pub fn dense(shape: Vec<usize>) -> Self {
        Self {
            shape,
            nnz: None,
            density: 1.0,
        }
    }

    /// Create stats for a sparse tensor
    pub fn sparse(shape: Vec<usize>, nnz: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let density = if total_elements > 0 {
            nnz as f64 / total_elements as f64
        } else {
            0.0
        };

        Self {
            shape,
            nnz: Some(nnz),
            density,
        }
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the effective number of non-zero elements
    pub fn effective_nnz(&self) -> usize {
        self.nnz.unwrap_or_else(|| self.size())
    }

    /// Check if tensor is sparse
    pub fn is_sparse(&self) -> bool {
        self.nnz.is_some() && self.density < 0.5
    }
}

/// Estimate FLOPs for a pairwise tensor contraction
///
/// # Arguments
///
/// * `spec` - Einsum specification for the contraction
/// * `stats` - Statistics for input tensors
///
/// # Returns
///
/// Estimated number of floating-point operations
///
/// # Complexity
///
/// For dense tensors: O(product of all unique indices)
/// For sparse tensors: O(nnz of inputs * size of contracted dimensions)
pub fn estimate_flops(spec: &EinsumSpec, stats: &[TensorStats]) -> Result<f64> {
    if spec.num_inputs() != stats.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match stats ({})",
            spec.num_inputs(),
            stats.len()
        ));
    }

    // Build dimension map: index -> size
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (input_spec, stat) in spec.inputs.iter().zip(stats.iter()) {
        if input_spec.len() != stat.shape.len() {
            return Err(anyhow!(
                "Input spec length ({}) does not match shape length ({})",
                input_spec.len(),
                stat.shape.len()
            ));
        }

        for (c, &size) in input_spec.chars().zip(stat.shape.iter()) {
            if let Some(&prev_size) = dim_map.get(&c) {
                if prev_size != size {
                    return Err(anyhow!(
                        "Dimension mismatch for index '{}': {} vs {}",
                        c,
                        prev_size,
                        size
                    ));
                }
            } else {
                dim_map.insert(c, size);
            }
        }
    }

    // Check if any input is sparse
    let has_sparse = stats.iter().any(|s| s.is_sparse());

    let flops = if has_sparse {
        // Sparse cost model: nnz(A) * nnz(B) * size(contracted dims) / size(all dims)
        estimate_sparse_flops(spec, stats, &dim_map)
    } else {
        // Dense cost model: output_size * contracted_size * 2 (multiply-add)
        // Output size: product of output indices
        let output_size: usize = spec
            .output
            .chars()
            .map(|c| *dim_map.get(&c).unwrap())
            .product::<usize>()
            .max(1); // Handle scalar output (empty output spec)

        // Contracted size: product of contracted indices
        let contracted_size: usize = spec
            .contracted_indices
            .iter()
            .map(|c| *dim_map.get(c).unwrap())
            .product::<usize>()
            .max(1); // Handle no contraction

        (output_size * contracted_size * 2) as f64
    };

    Ok(flops)
}

/// Estimate FLOPs for sparse tensor contraction
fn estimate_sparse_flops(
    spec: &EinsumSpec,
    stats: &[TensorStats],
    dim_map: &HashMap<char, usize>,
) -> f64 {
    // Simplified model: min(nnz of inputs) * size of output dims
    let min_nnz = stats.iter().map(|s| s.effective_nnz()).min().unwrap_or(0);

    let output_size: usize = spec
        .output
        .chars()
        .map(|c| *dim_map.get(&c).unwrap())
        .product();

    // Each output element may require multiple operations depending on nnz
    (min_nnz * output_size * 2) as f64
}

/// Estimate memory usage for a contraction
///
/// # Arguments
///
/// * `spec` - Einsum specification
/// * `stats` - Statistics for input tensors
/// * `bytes_per_element` - Size of each element (e.g., 8 for f64)
///
/// # Returns
///
/// Tuple of (input_memory, output_memory) in bytes
pub fn estimate_memory(
    spec: &EinsumSpec,
    stats: &[TensorStats],
    bytes_per_element: usize,
) -> Result<(usize, usize)> {
    if spec.num_inputs() != stats.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match stats ({})",
            spec.num_inputs(),
            stats.len()
        ));
    }

    // Input memory: sum of all input sizes
    let input_memory: usize = stats
        .iter()
        .map(|s| s.effective_nnz() * bytes_per_element)
        .sum();

    // Output memory: size of output tensor
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (input_spec, stat) in spec.inputs.iter().zip(stats.iter()) {
        for (c, &size) in input_spec.chars().zip(stat.shape.iter()) {
            dim_map.insert(c, size);
        }
    }

    let output_size: usize = spec
        .output
        .chars()
        .map(|c| *dim_map.get(&c).unwrap_or(&1))
        .product();

    let output_memory = output_size * bytes_per_element;

    Ok((input_memory, output_memory))
}

/// Estimate sparsity of output tensor
///
/// Uses a simple heuristic: output sparsity ≈ product of input densities
pub fn estimate_output_sparsity(spec: &EinsumSpec, stats: &[TensorStats]) -> Result<f64> {
    if spec.num_inputs() != stats.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match stats ({})",
            spec.num_inputs(),
            stats.len()
        ));
    }

    // Product of densities
    let mut density = 1.0;
    for stat in stats {
        density *= stat.density;
    }

    // Sparsity = 1 - density
    Ok(1.0 - density)
}

/// Complete cost estimate for a contraction
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Estimated FLOPs
    pub flops: f64,
    /// Input memory (bytes)
    pub input_memory: usize,
    /// Output memory (bytes)
    pub output_memory: usize,
    /// Total memory (bytes)
    pub total_memory: usize,
    /// Estimated output sparsity (0.0 = dense, 1.0 = all zeros)
    pub output_sparsity: f64,
}

impl CostEstimate {
    /// Compute a complete cost estimate
    pub fn compute(
        spec: &EinsumSpec,
        stats: &[TensorStats],
        bytes_per_element: usize,
    ) -> Result<Self> {
        let flops = estimate_flops(spec, stats)?;
        let (input_memory, output_memory) = estimate_memory(spec, stats, bytes_per_element)?;
        let output_sparsity = estimate_output_sparsity(spec, stats)?;

        Ok(Self {
            flops,
            input_memory,
            output_memory,
            total_memory: input_memory + output_memory,
            output_sparsity,
        })
    }

    /// Check if this contraction should use sparse representation
    pub fn should_use_sparse(&self) -> bool {
        // Use sparse if output sparsity > 90% and size is large
        self.output_sparsity > 0.9 && self.output_memory > 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_dense() {
        let stats = TensorStats::dense(vec![10, 20, 30]);
        assert_eq!(stats.size(), 6000);
        assert_eq!(stats.effective_nnz(), 6000);
        assert_eq!(stats.density, 1.0);
        assert!(!stats.is_sparse());
    }

    #[test]
    fn test_tensor_stats_sparse() {
        let stats = TensorStats::sparse(vec![100, 100, 100], 1000);
        assert_eq!(stats.size(), 1_000_000);
        assert_eq!(stats.effective_nnz(), 1000);
        assert_eq!(stats.density, 0.001);
        assert!(stats.is_sparse());
    }

    #[test]
    fn test_estimate_flops_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![
            TensorStats::dense(vec![10, 20]),
            TensorStats::dense(vec![20, 30]),
        ];

        let flops = estimate_flops(&spec, &stats).unwrap();

        // Matrix multiply: 10 * 30 output elements, each requires 20 multiply-adds
        // Total: 10 * 30 * 20 * 2 = 12,000 FLOPs
        assert_eq!(flops, 12_000.0);
    }

    #[test]
    fn test_estimate_flops_outer_product() {
        let spec = EinsumSpec::parse("i,j->ij").unwrap();
        let stats = vec![TensorStats::dense(vec![10]), TensorStats::dense(vec![20])];

        let flops = estimate_flops(&spec, &stats).unwrap();

        // Outer product: 10 * 20 = 200 output elements, no contraction
        // Total: 200 * 1 * 2 = 400 FLOPs
        assert_eq!(flops, 400.0);
    }

    #[test]
    fn test_estimate_flops_three_tensors() {
        let spec = EinsumSpec::parse("ijk,jkl,klm->ilm").unwrap();
        let stats = vec![
            TensorStats::dense(vec![5, 6, 7]),
            TensorStats::dense(vec![6, 7, 8]),
            TensorStats::dense(vec![7, 8, 9]),
        ];

        let flops = estimate_flops(&spec, &stats).unwrap();

        // Multi-input contraction cost model
        // Output indices: i=5, l=8, m=9
        // Output size: 5 * 8 * 9 = 360
        // Contracted indices: j=6, k=7
        // Contracted size: 6 * 7 = 42
        // FLOPs: 360 * 42 * 2 = 30,240
        //
        // Note: This estimates the cost of a single pairwise contraction step.
        // For actual multi-way contractions, the planner will sum costs across
        // the contraction tree.
        assert_eq!(flops, 30_240.0);
    }

    #[test]
    fn test_estimate_memory_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![
            TensorStats::dense(vec![10, 20]),
            TensorStats::dense(vec![20, 30]),
        ];

        let (input_mem, output_mem) = estimate_memory(&spec, &stats, 8).unwrap();

        // Input: 10*20 + 20*30 = 200 + 600 = 800 elements * 8 bytes = 6,400 bytes
        assert_eq!(input_mem, 6_400);

        // Output: 10*30 = 300 elements * 8 bytes = 2,400 bytes
        assert_eq!(output_mem, 2_400);
    }

    #[test]
    fn test_estimate_memory_sparse() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![
            TensorStats::sparse(vec![100, 100], 500), // 0.5% dense
            TensorStats::dense(vec![100, 100]),
        ];

        let (input_mem, output_mem) = estimate_memory(&spec, &stats, 8).unwrap();

        // Input: 500 (sparse) + 10,000 (dense) = 10,500 elements * 8 bytes = 84,000 bytes
        assert_eq!(input_mem, 84_000);

        // Output: 100*100 = 10,000 elements * 8 bytes = 80,000 bytes
        assert_eq!(output_mem, 80_000);
    }

    #[test]
    fn test_estimate_output_sparsity() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();

        // Two sparse tensors with 10% density each
        let stats = vec![
            TensorStats::sparse(vec![100, 100], 1_000), // 10% dense
            TensorStats::sparse(vec![100, 100], 1_000), // 10% dense
        ];

        let sparsity = estimate_output_sparsity(&spec, &stats).unwrap();

        // Output density ≈ 0.1 * 0.1 = 0.01 (1%)
        // Sparsity = 1 - 0.01 = 0.99 (99%)
        assert!((sparsity - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_cost_estimate_dense() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![
            TensorStats::dense(vec![10, 20]),
            TensorStats::dense(vec![20, 30]),
        ];

        let cost = CostEstimate::compute(&spec, &stats, 8).unwrap();

        assert_eq!(cost.flops, 12_000.0);
        assert_eq!(cost.input_memory, 6_400);
        assert_eq!(cost.output_memory, 2_400);
        assert_eq!(cost.total_memory, 8_800);
        assert!(!cost.should_use_sparse());
    }

    #[test]
    fn test_cost_estimate_sparse() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![
            TensorStats::sparse(vec![1000, 1000], 5_000), // 0.5% dense
            TensorStats::sparse(vec![1000, 1000], 10_000), // 1% dense
        ];

        let cost = CostEstimate::compute(&spec, &stats, 8).unwrap();

        // Output sparsity should be high
        assert!(cost.output_sparsity > 0.98);
        assert!(cost.should_use_sparse());
    }

    #[test]
    fn test_flops_error_mismatch() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![TensorStats::dense(vec![10, 20])]; // Only one tensor

        let result = estimate_flops(&spec, &stats);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_error_mismatch() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let stats = vec![TensorStats::dense(vec![10, 20])]; // Only one tensor

        let result = estimate_memory(&spec, &stats, 8);
        assert!(result.is_err());
    }
}
