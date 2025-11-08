//! Contraction order optimization
//!
//! Provides algorithms for finding optimal contraction orders for multi-tensor einsums.

use crate::api::{ContractionSpec, Plan, PlanHints, PlanNode, Planner, ReprHint};
use crate::cost::{estimate_flops, TensorStats};
use crate::parser::EinsumSpec;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Represents an intermediate tensor in the contraction sequence
#[derive(Debug, Clone)]
struct IntermediateTensor {
    /// Indices for this tensor (e.g., "ijk")
    indices: String,
    /// Shape of this tensor
    shape: Vec<usize>,
    /// Original input index (if this is an original input)
    original_idx: Option<usize>,
    /// Statistics for cost estimation
    stats: TensorStats,
}

impl IntermediateTensor {
    /// Create from an original input
    fn from_input(indices: String, shape: Vec<usize>, original_idx: usize) -> Self {
        let stats = TensorStats::dense(shape.clone());
        Self {
            indices,
            shape,
            original_idx: Some(original_idx),
            stats,
        }
    }

    /// Create an intermediate tensor from a contraction
    fn from_contraction(indices: String, shape: Vec<usize>, stats: TensorStats) -> Self {
        Self {
            indices,
            shape,
            original_idx: None,
            stats,
        }
    }
}

/// Find the contraction spec for two intermediate tensors
fn compute_pairwise_spec(a: &IntermediateTensor, b: &IntermediateTensor) -> Result<EinsumSpec> {
    // Build einsum spec: "a_indices,b_indices->output_indices"

    // Collect indices from both tensors
    let indices_a: std::collections::HashSet<char> = a.indices.chars().collect();
    let indices_b: std::collections::HashSet<char> = b.indices.chars().collect();

    // Find shared indices (these will be contracted)
    let shared: std::collections::HashSet<char> =
        indices_a.intersection(&indices_b).copied().collect();

    // Output indices: all indices EXCEPT the shared ones (which are contracted)
    let mut output_indices = Vec::new();

    // Add indices from a that are not shared
    for c in a.indices.chars() {
        if !shared.contains(&c) && !output_indices.contains(&c) {
            output_indices.push(c);
        }
    }

    // Add indices from b that are not shared
    for c in b.indices.chars() {
        if !shared.contains(&c) && !output_indices.contains(&c) {
            output_indices.push(c);
        }
    }

    // Sort for deterministic output
    output_indices.sort();
    let output: String = output_indices.into_iter().collect();

    let spec_str = format!("{},{}->{}", a.indices, b.indices, output);
    EinsumSpec::parse(&spec_str)
}

/// Compute the output shape for a pairwise contraction
fn compute_pairwise_output_shape(
    spec: &EinsumSpec,
    a: &IntermediateTensor,
    b: &IntermediateTensor,
) -> Result<Vec<usize>> {
    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();

    for (c, &size) in a.indices.chars().zip(a.shape.iter()) {
        dim_map.insert(c, size);
    }

    for (c, &size) in b.indices.chars().zip(b.shape.iter()) {
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

    // Build output shape
    let output_shape: Vec<usize> = spec
        .output
        .chars()
        .map(|c| *dim_map.get(&c).unwrap_or(&1))
        .collect();

    Ok(output_shape)
}

/// Greedy contraction order planner
///
/// Repeatedly contracts the pair of tensors with minimum cost until only one remains.
///
/// # Algorithm
///
/// 1. Start with all input tensors
/// 2. Find all possible pairwise contractions
/// 3. Estimate cost for each pair
/// 4. Contract the pair with minimum cost
/// 5. Repeat until only one tensor remains
///
/// # Complexity
///
/// O(n^3) where n is the number of inputs (n-1 steps, each checking O(n^2) pairs)
pub fn greedy_planner(
    spec: &EinsumSpec,
    shapes: &[Vec<usize>],
    _hints: &PlanHints,
) -> Result<Plan> {
    if spec.num_inputs() != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            spec.num_inputs(),
            shapes.len()
        ));
    }

    // Initialize intermediate tensors from inputs
    let mut intermediates: Vec<IntermediateTensor> = spec
        .inputs
        .iter()
        .zip(shapes.iter())
        .enumerate()
        .map(|(i, (indices, shape))| {
            IntermediateTensor::from_input(indices.clone(), shape.clone(), i)
        })
        .collect();

    let mut plan = Plan::new();
    let mut total_flops = 0.0;
    let mut peak_memory = 0;

    // Keep track of which intermediates correspond to which original inputs
    // This will be used to build the contraction order
    let mut contraction_order = Vec::new();

    // Contract until only one tensor remains
    while intermediates.len() > 1 {
        // Find the best pairwise contraction
        let mut best_cost = f64::INFINITY;
        let mut best_pair = (0, 1);
        let mut best_spec = None;
        let mut best_shape = None;

        for i in 0..intermediates.len() {
            for j in (i + 1)..intermediates.len() {
                let a = &intermediates[i];
                let b = &intermediates[j];

                // Compute contraction spec
                if let Ok(pairwise_spec) = compute_pairwise_spec(a, b) {
                    // Estimate cost
                    let stats = vec![a.stats.clone(), b.stats.clone()];
                    if let Ok(cost) = estimate_flops(&pairwise_spec, &stats) {
                        if cost < best_cost {
                            best_cost = cost;
                            best_pair = (i, j);

                            // Compute output shape
                            if let Ok(output_shape) =
                                compute_pairwise_output_shape(&pairwise_spec, a, b)
                            {
                                best_spec = Some(pairwise_spec);
                                best_shape = Some(output_shape);
                            }
                        }
                    }
                }
            }
        }

        if best_spec.is_none() || best_shape.is_none() {
            return Err(anyhow!("No valid contraction found"));
        }

        let pairwise_spec = best_spec.unwrap();
        let output_shape = best_shape.unwrap();

        // Record contraction in plan
        let (i, j) = best_pair;
        let a = &intermediates[i];
        let b = &intermediates[j];

        // Track original input indices for this contraction
        let input_indices = vec![
            a.original_idx.unwrap_or(1000 + i),
            b.original_idx.unwrap_or(1000 + j),
        ];

        let node = PlanNode {
            inputs: input_indices.clone(),
            output_spec: ContractionSpec::new(
                vec![a.indices.clone(), b.indices.clone()],
                pairwise_spec.output.clone(),
            ),
            cost: best_cost,
            memory: output_shape.iter().product::<usize>() * 8, // Assume f64
            repr: ReprHint::Auto,
        };

        plan.nodes.push(node);
        total_flops += best_cost;
        peak_memory = peak_memory.max(output_shape.iter().product::<usize>() * 8);

        // Track contraction order
        contraction_order.push(best_pair);

        // Create intermediate tensor from contraction result
        let output_stats = TensorStats::dense(output_shape.clone());
        let intermediate = IntermediateTensor::from_contraction(
            pairwise_spec.output.clone(),
            output_shape,
            output_stats,
        );

        // Remove contracted tensors (remove larger index first to avoid invalidation)
        let (remove_first, remove_second) = if i > j { (i, j) } else { (j, i) };
        intermediates.remove(remove_first);
        intermediates.remove(remove_second);

        // Add result
        intermediates.push(intermediate);
    }

    // Update plan with totals
    plan.estimated_flops = total_flops;
    plan.estimated_memory = peak_memory;
    plan.order = contraction_order;

    Ok(plan)
}

/// Dynamic programming planner for optimal contraction order
///
/// Uses bitmask DP to find the globally optimal contraction sequence.
/// Best for small tensor networks (< 20 tensors).
///
/// # Algorithm
///
/// For each subset S of tensors (represented as bitmask):
/// 1. Try all ways to partition S into two non-empty subsets S1 and S2
/// 2. Cost(S) = min over partitions of: Cost(S1) + Cost(S2) + cost(contract(S1, S2))
/// 3. Use memoization to avoid recomputation
/// 4. Backtrack to reconstruct optimal plan
///
/// # Complexity
///
/// O(3^n) time, O(2^n) space where n is number of inputs
/// Practical limit: n ≤ 20 tensors
///
/// # Fallback
///
/// Falls back to greedy planner if n > 20 (too expensive)
pub fn dp_planner(spec: &EinsumSpec, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
    let n = spec.num_inputs();

    // Fall back to greedy for large networks
    if n > 20 {
        log::warn!(
            "DP planner: too many inputs ({}), falling back to greedy",
            n
        );
        return greedy_planner(spec, shapes, hints);
    }

    if n != shapes.len() {
        return Err(anyhow!(
            "Number of inputs ({}) does not match shapes ({})",
            n,
            shapes.len()
        ));
    }

    // Handle trivial cases
    if n == 1 {
        return Ok(Plan::new()); // Single tensor, no contraction needed
    }

    if n == 2 {
        // Two tensors: direct contraction
        return greedy_planner(spec, shapes, hints);
    }

    // Initialize intermediates from inputs
    let intermediates: Vec<IntermediateTensor> = spec
        .inputs
        .iter()
        .zip(shapes.iter())
        .enumerate()
        .map(|(i, (indices, shape))| {
            IntermediateTensor::from_input(indices.clone(), shape.clone(), i)
        })
        .collect();

    // DP state: dp[mask] = (min_cost, best_partition)
    // mask is a bitmask representing a subset of tensors
    let num_states = 1 << n;
    let mut dp: Vec<Option<(f64, usize)>> = vec![None; num_states];

    // cached_contractions[mask] stores the result of contracting subset mask
    let mut cached_contractions: HashMap<usize, IntermediateTensor> = HashMap::new();

    // Base case: single tensors have zero contraction cost
    for (i, intermediate) in intermediates.iter().enumerate().take(n) {
        let mask = 1 << i;
        dp[mask] = Some((0.0, 0));
        cached_contractions.insert(mask, intermediate.clone());
    }

    // Fill DP table bottom-up (by increasing subset size)
    for mask in 1..num_states {
        let popcount = mask.count_ones() as usize;

        if popcount <= 1 {
            continue; // Already handled base case
        }

        let mut best_cost = f64::INFINITY;
        let mut best_partition = 0;

        // Try all ways to partition mask into two non-empty subsets
        // Iterate over all submasks of mask
        let mut submask = mask;
        loop {
            // Skip empty and full subsets
            if submask != 0 && submask != mask {
                let complement = mask ^ submask;

                // Check if both parts have been computed
                if let (Some((cost1, _)), Some((cost2, _))) = (dp[submask], dp[complement]) {
                    // Get cached contraction results
                    if let (Some(tensor1), Some(tensor2)) = (
                        cached_contractions.get(&submask),
                        cached_contractions.get(&complement),
                    ) {
                        // Compute cost of contracting these two intermediate results
                        if let Ok(pairwise_spec) = compute_pairwise_spec(tensor1, tensor2) {
                            let stats = vec![tensor1.stats.clone(), tensor2.stats.clone()];
                            if let Ok(contraction_cost) = estimate_flops(&pairwise_spec, &stats) {
                                let total_cost = cost1 + cost2 + contraction_cost;

                                if total_cost < best_cost {
                                    best_cost = total_cost;
                                    best_partition = submask;
                                }
                            }
                        }
                    }
                }
            }

            // Generate next submask (Gosper's hack variant)
            if submask == 0 {
                break;
            }
            submask = (submask - 1) & mask;
        }

        if best_partition == 0 {
            return Err(anyhow!(
                "DP planner: no valid partition found for mask {}",
                mask
            ));
        }

        dp[mask] = Some((best_cost, best_partition));

        // Cache the contraction result for this subset
        let submask = best_partition;
        let complement = mask ^ submask;

        if let (Some(tensor1), Some(tensor2)) = (
            cached_contractions.get(&submask),
            cached_contractions.get(&complement),
        ) {
            if let Ok(pairwise_spec) = compute_pairwise_spec(tensor1, tensor2) {
                if let Ok(output_shape) =
                    compute_pairwise_output_shape(&pairwise_spec, tensor1, tensor2)
                {
                    let output_stats = TensorStats::dense(output_shape.clone());
                    let intermediate = IntermediateTensor::from_contraction(
                        pairwise_spec.output.clone(),
                        output_shape,
                        output_stats,
                    );
                    cached_contractions.insert(mask, intermediate);
                }
            }
        }
    }

    // Reconstruct the plan by backtracking through DP table
    let full_mask = (1 << n) - 1;
    let (total_cost, _) = dp[full_mask].ok_or_else(|| anyhow!("DP planner: no solution found"))?;

    let mut plan = Plan::new();
    plan.estimated_flops = total_cost;

    // Recursively build plan nodes by following optimal partitions
    fn reconstruct_plan(
        mask: usize,
        dp: &[Option<(f64, usize)>],
        cached_contractions: &HashMap<usize, IntermediateTensor>,
        plan: &mut Plan,
        peak_memory: &mut usize,
    ) -> Result<()> {
        if mask.count_ones() == 1 {
            // Base case: single tensor
            return Ok(());
        }

        let (_, best_partition) =
            dp[mask].ok_or_else(|| anyhow!("Missing DP state for mask {}", mask))?;
        let submask = best_partition;
        let complement = mask ^ submask;

        // Recursively process left and right subtrees
        reconstruct_plan(submask, dp, cached_contractions, plan, peak_memory)?;
        reconstruct_plan(complement, dp, cached_contractions, plan, peak_memory)?;

        // Add the contraction node for this step
        let tensor1 = cached_contractions
            .get(&submask)
            .ok_or_else(|| anyhow!("Missing cached tensor for submask {}", submask))?;
        let tensor2 = cached_contractions
            .get(&complement)
            .ok_or_else(|| anyhow!("Missing cached tensor for complement {}", complement))?;

        let pairwise_spec = compute_pairwise_spec(tensor1, tensor2)?;
        let output_shape = compute_pairwise_output_shape(&pairwise_spec, tensor1, tensor2)?;

        let stats = vec![tensor1.stats.clone(), tensor2.stats.clone()];
        let cost = estimate_flops(&pairwise_spec, &stats)?;
        let memory = output_shape.iter().product::<usize>() * 8;

        *peak_memory = (*peak_memory).max(memory);

        // Build input indices list
        let mut input_indices = Vec::new();
        for i in 0..64 {
            if (submask & (1 << i)) != 0 {
                input_indices.push(i);
            }
        }
        for i in 0..64 {
            if (complement & (1 << i)) != 0 {
                input_indices.push(i);
            }
        }

        let node = PlanNode {
            inputs: input_indices,
            output_spec: ContractionSpec::new(
                vec![tensor1.indices.clone(), tensor2.indices.clone()],
                pairwise_spec.output.clone(),
            ),
            cost,
            memory,
            repr: ReprHint::Auto,
        };

        plan.nodes.push(node);
        Ok(())
    }

    let mut peak_memory = 0;
    reconstruct_plan(
        full_mask,
        &dp,
        &cached_contractions,
        &mut plan,
        &mut peak_memory,
    )?;

    plan.estimated_memory = peak_memory;

    Ok(plan)
}

// ============================================================================
// Planner Trait Implementations
// ============================================================================

/// Greedy contraction order planner (struct-based)
///
/// Implements the [`Planner`] trait using a greedy heuristic.
///
/// # Example
///
/// ```
/// use tenrso_planner::{GreedyPlanner, Planner, PlanHints};
///
/// let planner = GreedyPlanner::new();
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk->ik",
///     &[vec![10, 20], vec![20, 30]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GreedyPlanner;

impl GreedyPlanner {
    /// Create a new greedy planner
    pub fn new() -> Self {
        Self
    }
}

impl Planner for GreedyPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        greedy_planner(&parsed_spec, shapes, hints)
    }
}

/// Dynamic programming contraction order planner (struct-based)
///
/// Implements the [`Planner`] trait using optimal DP algorithm.
/// Automatically falls back to greedy for large networks (n > 20).
///
/// # Example
///
/// ```
/// use tenrso_planner::{DPPlanner, Planner, PlanHints};
///
/// let planner = DPPlanner::new();
/// let hints = PlanHints::default();
/// let plan = planner.make_plan(
///     "ij,jk,kl->il",
///     &[vec![10, 20], vec![20, 30], vec![30, 10]],
///     &hints
/// ).unwrap();
///
/// assert_eq!(plan.nodes.len(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DPPlanner;

impl DPPlanner {
    /// Create a new DP planner
    pub fn new() -> Self {
        Self
    }
}

impl Planner for DPPlanner {
    fn make_plan(&self, spec: &str, shapes: &[Vec<usize>], hints: &PlanHints) -> Result<Plan> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        dp_planner(&parsed_spec, shapes, hints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_planner_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        // Should have 1 contraction step (2 inputs -> 1 output)
        assert_eq!(plan.nodes.len(), 1);

        // Check estimated FLOPs
        assert!(plan.estimated_flops > 0.0);

        // Check contraction order
        assert_eq!(plan.order.len(), 1);
    }

    #[test]
    fn test_greedy_planner_three_tensors() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15], vec![15, 20]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        // Should have 2 contraction steps (3 inputs -> 2 -> 1)
        assert_eq!(plan.nodes.len(), 2);

        // FLOPs should be sum of both steps
        assert!(plan.estimated_flops > 0.0);

        // Should have 2 contractions recorded
        assert_eq!(plan.order.len(), 2);
    }

    #[test]
    fn test_greedy_planner_large_chain() {
        // Chain contraction: A_ij * B_jk * C_kl * D_lm -> A_im
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 50]];
        let hints = PlanHints::default();

        let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        // Should have 3 contraction steps
        assert_eq!(plan.nodes.len(), 3);

        // Check that we have a valid plan
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }

    #[test]
    fn test_greedy_planner_error_shape_mismatch() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20]]; // Only 1 shape for 2 inputs

        let hints = PlanHints::default();
        let result = greedy_planner(&spec, &shapes, &hints);

        assert!(result.is_err());
    }

    #[test]
    fn test_compute_pairwise_spec() {
        let a = IntermediateTensor::from_input("ij".to_string(), vec![10, 20], 0);
        let b = IntermediateTensor::from_input("jk".to_string(), vec![20, 30], 1);

        let spec = compute_pairwise_spec(&a, &b).unwrap();

        // Output should contain non-shared indices: i, k
        // Shared index j should be contracted (not in output)
        assert!(spec.output.contains('i'));
        assert!(!spec.output.contains('j')); // j is contracted
        assert!(spec.output.contains('k'));
        assert_eq!(spec.output.len(), 2); // Only i and k
    }

    #[test]
    fn test_compute_pairwise_output_shape() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let a = IntermediateTensor::from_input("ij".to_string(), vec![10, 20], 0);
        let b = IntermediateTensor::from_input("jk".to_string(), vec![20, 30], 1);

        let shape = compute_pairwise_output_shape(&spec, &a, &b).unwrap();

        // Output shape should be [10, 30] for indices i, k (j is contracted)
        assert_eq!(shape, vec![10, 30]);
    }

    // ============================================================================
    // Dynamic Programming Planner Tests
    // ============================================================================

    #[test]
    fn test_dp_planner_matmul() {
        let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 1 contraction step (2 inputs -> 1 output)
        assert_eq!(plan.nodes.len(), 1);

        // Check estimated FLOPs (should be 10 * 20 * 30 * 2 = 12000)
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }

    #[test]
    fn test_dp_planner_three_tensors() {
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![5, 10], vec![10, 15], vec![15, 20]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 2 contraction steps (3 inputs -> 2 -> 1)
        assert_eq!(plan.nodes.len(), 2);

        // FLOPs should be positive
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }

    #[test]
    fn test_dp_planner_four_tensors() {
        // Chain: A_ij * B_jk * C_kl * D_lm -> A_im
        let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
        let shapes = vec![vec![10, 20], vec![20, 30], vec![30, 40], vec![40, 50]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 3 contraction steps
        assert_eq!(plan.nodes.len(), 3);

        // Check that we have a valid plan with reasonable costs
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }

    #[test]
    fn test_dp_planner_single_tensor() {
        // Single tensor: no contraction needed
        let spec = EinsumSpec::parse("ijk->ijk").unwrap();
        let shapes = vec![vec![10, 20, 30]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // No contraction steps needed
        assert_eq!(plan.nodes.len(), 0);
        assert_eq!(plan.estimated_flops, 0.0);
    }

    #[test]
    fn test_dp_planner_vs_greedy_small() {
        // For small problems, DP should find optimal or better solution than greedy
        let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
        let shapes = vec![vec![100, 10], vec![10, 100], vec![100, 10]];
        let hints = PlanHints::default();

        let dp_plan = dp_planner(&spec, &shapes, &hints).unwrap();
        let greedy_plan = greedy_planner(&spec, &shapes, &hints).unwrap();

        // Both should succeed
        assert_eq!(dp_plan.nodes.len(), 2);
        assert_eq!(greedy_plan.nodes.len(), 2);

        // DP should find optimal cost (may be equal to greedy for this case)
        // The key is that DP cost should never be worse than greedy
        assert!(dp_plan.estimated_flops <= greedy_plan.estimated_flops * 1.01); // Allow 1% tolerance
    }

    #[test]
    fn test_dp_planner_star_contraction() {
        // Star contraction: all tensors share a common index
        let spec = EinsumSpec::parse("ia,ib,ic->iabc").unwrap();
        let shapes = vec![vec![10, 5], vec![10, 6], vec![10, 7]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 2 contraction steps
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
    }

    #[test]
    fn test_dp_planner_fallback_large() {
        // Test fallback to greedy for large n
        // Create a spec with 21 inputs (over limit)
        // Use single-character indices a-z (26 letters available)
        let indices = "abcdefghijklmnopqrstu"; // 21 characters
        let mut input_specs = Vec::new();
        let mut shapes = Vec::new();

        for c in indices.chars() {
            input_specs.push(c.to_string());
            shapes.push(vec![2]); // Small shapes
        }

        // All tensors contract to produce output with first index
        let spec_str = format!("{}->{}", input_specs.join(","), input_specs[0]);
        let spec = EinsumSpec::parse(&spec_str).unwrap();
        let hints = PlanHints::default();

        // Should fall back to greedy without error
        let result = dp_planner(&spec, &shapes, &hints);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dp_planner_inner_product() {
        // Inner product: ij,ij->
        let spec = EinsumSpec::parse("ij,ij->").unwrap();
        let shapes = vec![vec![10, 20], vec![10, 20]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 1 contraction (element-wise multiply + sum)
        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }

    #[test]
    fn test_dp_planner_outer_product() {
        // Outer product: i,j,k->ijk
        let spec = EinsumSpec::parse("i,j,k->ijk").unwrap();
        let shapes = vec![vec![10], vec![20], vec![30]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 2 contraction steps
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.estimated_flops > 0.0);
    }

    #[test]
    fn test_dp_planner_five_tensors() {
        // Test with 5 tensors to stress DP
        let spec = EinsumSpec::parse("ab,bc,cd,de,ea->abcde").unwrap();
        let shapes = vec![vec![5, 6], vec![6, 7], vec![7, 8], vec![8, 9], vec![9, 5]];
        let hints = PlanHints::default();

        let plan = dp_planner(&spec, &shapes, &hints).unwrap();

        // Should have 4 contraction steps
        assert_eq!(plan.nodes.len(), 4);
        assert!(plan.estimated_flops > 0.0);
        assert!(plan.estimated_memory > 0);
    }

    // ============================================================================
    // Planner Trait Implementation Tests
    // ============================================================================

    #[test]
    fn test_greedy_planner_struct() {
        let planner = GreedyPlanner::new();
        let hints = PlanHints::default();

        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();

        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }

    #[test]
    fn test_greedy_planner_default() {
        let planner = GreedyPlanner;
        let hints = PlanHints::default();

        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![5, 10], vec![10, 15], vec![15, 20]],
                &hints,
            )
            .unwrap();

        assert_eq!(plan.nodes.len(), 2);
    }

    #[test]
    fn test_dp_planner_struct() {
        let planner = DPPlanner::new();
        let hints = PlanHints::default();

        let plan = planner
            .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
            .unwrap();

        assert_eq!(plan.nodes.len(), 1);
        assert!(plan.estimated_flops > 0.0);
    }

    #[test]
    fn test_dp_planner_default() {
        let planner = DPPlanner;
        let hints = PlanHints::default();

        let plan = planner
            .make_plan(
                "ij,jk,kl->il",
                &[vec![5, 10], vec![10, 15], vec![15, 20]],
                &hints,
            )
            .unwrap();

        assert_eq!(plan.nodes.len(), 2);
    }

    #[test]
    fn test_planner_trait_polymorphism() {
        fn test_planner(planner: &dyn Planner) {
            let hints = PlanHints::default();
            let plan = planner
                .make_plan("ij,jk->ik", &[vec![10, 20], vec![20, 30]], &hints)
                .unwrap();
            assert_eq!(plan.nodes.len(), 1);
        }

        test_planner(&GreedyPlanner::new());
        test_planner(&DPPlanner::new());
    }

    // ============================================================================
    // Property-Based Tests
    // ============================================================================

    use proptest::prelude::*;

    proptest! {
        /// Property: Plans should always succeed for valid matmul specs
        #[test]
        fn prop_greedy_planner_matmul_always_succeeds(
            shared_dim in 1..=50usize,
            m in 1..=30usize,
            n in 1..=30usize,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![m, shared_dim], vec![shared_dim, n]];
            let hints = PlanHints::default();

            let result = greedy_planner(&spec, &shapes, &hints);
            prop_assert!(result.is_ok());

            let plan = result.unwrap();
            prop_assert_eq!(plan.nodes.len(), 1);
            prop_assert!(plan.estimated_flops > 0.0);
            prop_assert!(plan.estimated_memory > 0);
        }

        /// Property: DP planner should find equal or better cost than greedy
        #[test]
        fn prop_dp_cost_less_or_equal_greedy(
            dims in prop::collection::vec(2..=10usize, 3..=3)
        ) {
            // Create a chain contraction: ij,jk,kl->il
            let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
            let shapes = vec![
                vec![dims[0], dims[1]],
                vec![dims[1], dims[2]],
                vec![dims[2], dims[0]],
            ];
            let hints = PlanHints::default();

            let greedy_result = greedy_planner(&spec, &shapes, &hints);
            let dp_result = dp_planner(&spec, &shapes, &hints);

            if greedy_result.is_ok() && dp_result.is_ok() {
                let greedy_plan = greedy_result.unwrap();
                let dp_plan = dp_result.unwrap();

                // DP should be optimal, so cost ≤ greedy cost
                prop_assert!(
                    dp_plan.estimated_flops <= greedy_plan.estimated_flops * 1.01,
                    "DP cost {} should be ≤ greedy cost {}",
                    dp_plan.estimated_flops,
                    greedy_plan.estimated_flops
                );
            }
        }

        /// Property: Number of contraction steps should be n-1 for n inputs
        #[test]
        fn prop_contraction_steps_count(n_inputs in 2..=6usize) {
            // Create n tensors with compatible shapes
            let mut input_specs = Vec::new();
            let mut shapes = Vec::new();

            // Create a chain: a,ab,bc,cd,...
            let indices: Vec<char> = "abcdefghij".chars().collect();

            for i in 0..n_inputs {
                let idx1 = indices[i];
                let idx2 = if i < n_inputs - 1 { indices[i + 1] } else { indices[0] };
                input_specs.push(format!("{}{}", idx1, idx2));
                shapes.push(vec![5, 6]);
            }

            // Output is first and last index
            let output = format!("{}{}", indices[0], indices[n_inputs - 1]);
            let spec_str = format!("{}->{}", input_specs.join(","), output);

            if let Ok(spec) = EinsumSpec::parse(&spec_str) {
                let hints = PlanHints::default();

                if let Ok(plan) = greedy_planner(&spec, &shapes, &hints) {
                    // Should have exactly n-1 contraction steps
                    prop_assert_eq!(
                        plan.nodes.len(),
                        n_inputs - 1,
                        "Expected {} contraction steps for {} inputs",
                        n_inputs - 1,
                        n_inputs
                    );
                }
            }
        }

        /// Property: Plans should be deterministic
        #[test]
        fn prop_plans_are_deterministic(
            m in 5..=20usize,
            n in 5..=20usize,
            k in 5..=20usize,
        ) {
            let spec = EinsumSpec::parse("ij,jk,kl->il").unwrap();
            let shapes = vec![vec![m, n], vec![n, k], vec![k, m]];
            let hints = PlanHints::default();

            let plan1 = greedy_planner(&spec, &shapes, &hints).unwrap();
            let plan2 = greedy_planner(&spec, &shapes, &hints).unwrap();

            // Plans should be identical
            prop_assert_eq!(plan1.nodes.len(), plan2.nodes.len());
            prop_assert_eq!(plan1.estimated_flops, plan2.estimated_flops);
            prop_assert_eq!(plan1.estimated_memory, plan2.estimated_memory);
        }

        /// Property: Peak memory should be at least as large as output
        #[test]
        fn prop_peak_memory_exceeds_output(
            m in 10..=30usize,
            n in 10..=30usize,
        ) {
            let spec = EinsumSpec::parse("ij,jk->ik").unwrap();
            let shapes = vec![vec![m, n], vec![n, m]];
            let hints = PlanHints::default();

            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

            // Output size is m * m * 8 bytes (f64)
            let output_size = m * m * 8;

            prop_assert!(
                plan.estimated_memory >= output_size,
                "Peak memory {} should be >= output size {}",
                plan.estimated_memory,
                output_size
            );
        }

        /// Property: Empty plans for single tensors
        #[test]
        fn prop_single_tensor_no_contraction(rank in 1..=4usize, size in 5..=20usize) {
            // Single tensor with uniform dimensions
            let indices: String = "ijkl".chars().take(rank).collect();
            let spec_str = format!("{}->{}", indices, indices);

            if let Ok(spec) = EinsumSpec::parse(&spec_str) {
                let shapes = vec![vec![size; rank]];
                let hints = PlanHints::default();

                let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

                // No contraction needed for single tensor
                prop_assert_eq!(plan.nodes.len(), 0);
                prop_assert_eq!(plan.estimated_flops, 0.0);
            }
        }

        /// Property: All nodes should have positive cost
        #[test]
        fn prop_all_nodes_positive_cost(
            dims in prop::collection::vec(5..=15usize, 4..=4)
        ) {
            let spec = EinsumSpec::parse("ij,jk,kl,lm->im").unwrap();
            let shapes = vec![
                vec![dims[0], dims[1]],
                vec![dims[1], dims[2]],
                vec![dims[2], dims[3]],
                vec![dims[3], dims[0]],
            ];
            let hints = PlanHints::default();

            let plan = greedy_planner(&spec, &shapes, &hints).unwrap();

            // All nodes should have positive cost
            for node in &plan.nodes {
                prop_assert!(
                    node.cost > 0.0,
                    "All contraction costs should be positive, found {}",
                    node.cost
                );
            }
        }
    }
}
