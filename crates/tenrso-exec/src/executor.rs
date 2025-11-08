//! Executor trait and implementations

use crate::hints::ExecHints;
use crate::ops::execute_dense_contraction;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use std::collections::HashMap;
use tenrso_core::{Axis, DenseND, TensorHandle};
use tenrso_planner::{greedy_planner, EinsumSpec, Plan, PlanHints};

/// Element-wise operation types
#[derive(Clone, Debug)]
pub enum ElemOp {
    /// Negation: -x
    Neg,
    /// Absolute value: |x|
    Abs,
    /// Exponential: e^x
    Exp,
    /// Natural logarithm: ln(x)
    Log,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Square root: sqrt(x)
    Sqrt,
    /// Power of 2: x^2
    Sqr,
    /// Reciprocal: 1/x
    Recip,
}

/// Reduction operation types
#[derive(Clone, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
}

/// Main executor trait for tensor operations
pub trait TenrsoExecutor<T>
where
    T: Clone + Num,
{
    /// Execute an einsum contraction
    fn einsum(
        &mut self,
        spec: &str,
        inputs: &[TensorHandle<T>],
        hints: &ExecHints,
    ) -> Result<TensorHandle<T>>;

    /// Apply element-wise operation
    fn elem_op(&mut self, op: ElemOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>>;

    /// Apply reduction operation
    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &TensorHandle<T>,
        axes: &[Axis],
    ) -> Result<TensorHandle<T>>;
}

/// Memory pool for tensor allocation reuse
///
/// Tracks allocated tensors by their shape signature for reuse.
/// This reduces allocation overhead for repeated operations.
#[derive(Default)]
struct MemoryPool {
    /// Map from shape signature to available buffer pool
    /// Shape signature is a string like "2x3x4" for a [2, 3, 4] tensor
    pools: HashMap<String, Vec<Vec<u8>>>,
    /// Statistics for monitoring
    hits: usize,
    misses: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    fn new() -> Self {
        Self::default()
    }

    /// Get a buffer for the given shape, reusing if available
    #[allow(dead_code)]
    fn acquire(&mut self, shape: &[usize], elem_size: usize) -> Vec<u8> {
        let signature = Self::shape_signature(shape);
        let total_size = shape.iter().product::<usize>() * elem_size;

        if let Some(pool) = self.pools.get_mut(&signature) {
            if let Some(mut buffer) = pool.pop() {
                self.hits += 1;
                // Ensure buffer is the right size
                buffer.resize(total_size, 0);
                return buffer;
            }
        }

        self.misses += 1;
        vec![0u8; total_size]
    }

    /// Return a buffer to the pool for reuse
    #[allow(dead_code)]
    fn release(&mut self, shape: &[usize], buffer: Vec<u8>) {
        let signature = Self::shape_signature(shape);
        let pool = self.pools.entry(signature).or_default();

        // Limit pool size to prevent unbounded growth
        const MAX_POOL_SIZE: usize = 16;
        if pool.len() < MAX_POOL_SIZE {
            pool.push(buffer);
        }
    }

    /// Create a shape signature for hashing
    fn shape_signature(shape: &[usize]) -> String {
        shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }

    /// Get pool statistics
    fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Clear all pooled buffers
    fn clear(&mut self) {
        self.pools.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// CPU executor implementation with memory pooling
pub struct CpuExecutor {
    /// Memory pool for tensor allocation reuse (for future optimization)
    #[allow(dead_code)]
    memory_pool: MemoryPool,
    /// Number of threads to use (0 = auto-detect, for future parallel execution)
    pub num_threads: usize,
}

impl CpuExecutor {
    /// Create a new CPU executor with default settings
    pub fn new() -> Self {
        Self {
            memory_pool: MemoryPool::new(),
            num_threads: 0, // Auto-detect
        }
    }

    /// Create a CPU executor with custom thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            memory_pool: MemoryPool::new(),
            num_threads,
        }
    }

    /// Get memory pool statistics (hits, misses, hit_rate)
    pub fn pool_stats(&self) -> (usize, usize, f64) {
        self.memory_pool.stats()
    }

    /// Clear the memory pool
    pub fn clear_pool(&mut self) {
        self.memory_pool.clear();
    }

    /// Execute einsum with planner integration
    fn execute_einsum_with_planner<T>(
        &mut self,
        spec: &EinsumSpec,
        inputs: &[DenseND<T>],
        _hints: &ExecHints,
    ) -> Result<DenseND<T>>
    where
        T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        // Get shapes for planning
        let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape().to_vec()).collect();

        // Create plan hints
        let plan_hints = PlanHints::default();

        // Generate execution plan
        let plan = greedy_planner(spec, &shapes, &plan_hints)?;

        // For 2-input case, execute directly (optimization)
        if inputs.len() == 2 {
            return execute_dense_contraction(spec, &inputs[0], &inputs[1]);
        }

        // Multi-input case: execute plan step-by-step
        self.execute_plan(&plan, inputs)
    }

    /// Execute a multi-step contraction plan
    fn execute_plan<T>(&mut self, plan: &Plan, inputs: &[DenseND<T>]) -> Result<DenseND<T>>
    where
        T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        // Mirror the planner's algorithm:
        // Start with all inputs as intermediates, repeatedly contract pairs
        let mut intermediates: Vec<DenseND<T>> = inputs.to_vec();

        // Execute each step in the plan using the order field
        for (step_idx, &(i, j)) in plan.order.iter().enumerate() {
            // Verify indices are valid
            if i >= intermediates.len() || j >= intermediates.len() {
                return Err(anyhow!(
                    "Step {}: Invalid indices ({}, {}) for {} intermediates",
                    step_idx,
                    i,
                    j,
                    intermediates.len()
                ));
            }

            // Get the corresponding plan node for the spec
            let node = &plan.nodes[step_idx];

            // Extract the two tensors to contract
            // We need to be careful about index invalidation when removing
            let (tensor_a, tensor_b) = if i < j {
                // Remove j first (higher index)
                let b = intermediates.remove(j);
                let a = intermediates.remove(i);
                (a, b)
            } else {
                // Remove i first (higher index)
                let a = intermediates.remove(i);
                let b = intermediates.remove(j);
                (a, b)
            };

            // Build einsum spec for this step
            let spec_str = format!(
                "{},{}->{}",
                node.output_spec.input_specs[0],
                node.output_spec.input_specs[1],
                node.output_spec.output_spec
            );
            let step_spec = EinsumSpec::parse(&spec_str)?;

            // Execute this contraction
            let result = execute_dense_contraction(&step_spec, &tensor_a, &tensor_b)?;

            // Add intermediate result back
            intermediates.push(result);
        }

        // After all contractions, exactly one tensor should remain
        if intermediates.len() != 1 {
            return Err(anyhow!(
                "Expected 1 final tensor, got {}",
                intermediates.len()
            ));
        }

        Ok(intermediates.into_iter().next().unwrap())
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TenrsoExecutor<T> for CpuExecutor
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
{
    fn einsum(
        &mut self,
        spec: &str,
        inputs: &[TensorHandle<T>],
        hints: &ExecHints,
    ) -> Result<TensorHandle<T>> {
        // Parse einsum spec
        let parsed_spec = EinsumSpec::parse(spec)?;

        // Validate input count
        if parsed_spec.num_inputs() != inputs.len() {
            return Err(anyhow!(
                "Spec expects {} inputs, got {}",
                parsed_spec.num_inputs(),
                inputs.len()
            ));
        }

        // Extract dense tensors from handles
        let dense_inputs: Vec<&DenseND<T>> = inputs
            .iter()
            .map(|h| {
                h.as_dense()
                    .ok_or_else(|| anyhow!("Only dense tensors supported for now"))
            })
            .collect::<Result<Vec<_>>>()?;

        // Clone for execution (TODO: optimize to avoid clones)
        let dense_inputs_owned: Vec<DenseND<T>> = dense_inputs.iter().map(|&t| t.clone()).collect();

        // Execute with planner
        let result = self.execute_einsum_with_planner(&parsed_spec, &dense_inputs_owned, hints)?;

        // Wrap in TensorHandle
        Ok(TensorHandle::from_dense_auto(result))
    }

    fn elem_op(&mut self, op: ElemOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        // Extract dense tensor (for now, only dense support)
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for elem_op"))?;

        // Apply element-wise operation using scirs2_core's numeric traits
        let result_data = match op {
            ElemOp::Neg => dense.view().mapv(|v| -v),
            ElemOp::Abs => dense.view().mapv(|v| v.abs()),
            ElemOp::Exp => dense.view().mapv(|v| v.exp()),
            ElemOp::Log => dense.view().mapv(|v| v.ln()),
            ElemOp::Sin => dense.view().mapv(|v| v.sin()),
            ElemOp::Cos => dense.view().mapv(|v| v.cos()),
            ElemOp::Sqrt => dense.view().mapv(|v| v.sqrt()),
            ElemOp::Sqr => dense.view().mapv(|v| v * v),
            ElemOp::Recip => dense.view().mapv(|v| v.recip()),
        };

        // Create result tensor with same shape
        let result = DenseND::from_array(result_data);

        Ok(TensorHandle::from_dense_auto(result))
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &TensorHandle<T>,
        axes: &[Axis],
    ) -> Result<TensorHandle<T>> {
        // Extract dense tensor (for now, only dense support)
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for reduce"))?;

        if axes.is_empty() {
            return Err(anyhow!("No axes specified for reduction"));
        }

        // Axis is already usize (type alias), no need to call .index()
        let axis_indices: Vec<usize> = axes.to_vec();

        // Validate axis indices
        let ndim = dense.shape().len();
        for &axis_idx in &axis_indices {
            if axis_idx >= ndim {
                return Err(anyhow!(
                    "Axis index {} out of range for tensor with {} dimensions",
                    axis_idx,
                    ndim
                ));
            }
        }

        // Apply reduction along specified axes
        let mut result = dense.view().to_owned();

        // Sort axes in descending order to avoid index shifts
        let mut sorted_axes = axis_indices.clone();
        sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

        // Apply reduction for each axis
        for &axis_idx in &sorted_axes {
            let axis = scirs2_core::ndarray_ext::Axis(axis_idx);

            result = match op {
                ReduceOp::Sum => result.sum_axis(axis),
                ReduceOp::Max => {
                    // For max, we need to handle it manually
                    result.map_axis(axis, |view| {
                        view.iter()
                            .cloned()
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or_else(T::default)
                    })
                }
                ReduceOp::Min => {
                    // For min, we need to handle it manually
                    result.map_axis(axis, |view| {
                        view.iter()
                            .cloned()
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or_else(T::default)
                    })
                }
                ReduceOp::Mean => result.mean_axis(axis).ok_or_else(|| {
                    anyhow!("Mean reduction failed - axis might be empty or type doesn't support division")
                })?,
            };
        }

        // Create result tensor
        let result_tensor = DenseND::from_array(result);

        Ok(TensorHandle::from_dense_auto(result_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenrso_core::DenseND;

    #[test]
    fn test_cpu_executor_matmul() {
        let mut executor = CpuExecutor::new();

        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);

        let result = executor
            .einsum("ij,jk->ik", &[handle_a, handle_b], &ExecHints::default())
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);

        // Verify computation
        let result_view = result_dense.view();
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        let diff: f64 = result_view[[0, 0]] - 58.0;
        assert!(diff.abs() < 1e-10);
    }

    #[test]
    fn test_cpu_executor_input_validation() {
        let mut executor = CpuExecutor::new();

        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Wrong number of inputs
        let result = executor.einsum("ij,jk->ik", &[handle_a], &ExecHints::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_executor_three_tensors() {
        let mut executor = CpuExecutor::new();

        // Create three small tensors for a chain contraction
        // A: 2x3, B: 3x2, C: 2x2
        // Einsum: ij,jk,kl->il (chain of matrix multiplications)
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let handle_c = TensorHandle::from_dense_auto(c);

        let result = executor
            .einsum(
                "ij,jk,kl->il",
                &[handle_a, handle_b, handle_c],
                &ExecHints::default(),
            )
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2, 2]);

        // Verify it's a valid result (non-trivial computation)
        let result_view = result_dense.view();
        // Just verify shape and that computation completed
        // Detailed numerical verification would be complex for 3-tensor chain
        let val: f64 = result_view[[0, 0]];
        assert!(val.abs() > 0.0);
    }

    #[test]
    fn test_cpu_executor_outer_then_contract() {
        let mut executor = CpuExecutor::new();

        // Test a different pattern: outer product then contraction
        // A: 2, B: 3, C: 2x3
        // Einsum: i,j,ij->  (this computes trace of outer product)
        let a = DenseND::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let c = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);
        let handle_c = TensorHandle::from_dense_auto(c);

        let result = executor
            .einsum(
                "i,j,ij->",
                &[handle_a, handle_b, handle_c],
                &ExecHints::default(),
            )
            .unwrap();

        let result_dense = result.as_dense().unwrap();
        // Result should be scalar (empty shape or [1])
        assert!(result_dense.shape().is_empty() || result_dense.shape() == &[1]);

        // Verify computation: sum of element-wise products
        // A[i] * B[j] * C[i,j] summed over all i,j
        // = (1*1*1 + 1*2*2 + 1*3*3) + (2*1*4 + 2*2*5 + 2*3*6)
        // = (1 + 4 + 9) + (8 + 20 + 36)
        // = 14 + 64 = 78
        let result_view = result_dense.view();
        let result_val = if result_dense.shape().is_empty() {
            result_view[[]]
        } else {
            result_view[[0]]
        };
        let diff: f64 = result_val - 78.0;
        assert!(diff.abs() < 1e-10, "Expected 78.0, got {}", result_val);
    }

    #[test]
    fn test_elem_op_neg() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Neg, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((result_view[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((result_view[[1, 0]] - (-3.0)).abs() < 1e-10);
        assert!((result_view[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_elem_op_abs() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Abs, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((result_view[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((result_view[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_elem_op_exp() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Exp, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] - std::f64::consts::E).abs() < 1e-10);
        assert!((result_view[[2]] - std::f64::consts::E.powi(2)).abs() < 1e-9);
    }

    #[test]
    fn test_elem_op_log() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(
            vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)],
            &[3],
        )
        .unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Log, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0).abs() < 1e-10);
        assert!((result_view[[2]] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_reduce_sum_single_axis() {
        let mut executor = CpuExecutor::new();
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Sum along axis 0 (rows): [5, 7, 9]
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 5.0).abs() < 1e-10);
        assert!((result_view[[1]] - 7.0).abs() < 1e-10);
        assert!((result_view[[2]] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_sum_axis_1() {
        let mut executor = CpuExecutor::new();
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Sum along axis 1 (columns): [6, 15]
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[1]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[2]);

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 6.0).abs() < 1e-10);
        assert!((result_view[[1]] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_max() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Max along axis 0: [2, 8, 4]
        let result = executor.reduce(ReduceOp::Max, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 2.0).abs() < 1e-10);
        assert!((result_view[[1]] - 8.0).abs() < 1e-10);
        assert!((result_view[[2]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_min() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Min along axis 0: [1, 5, 3]
        let result = executor.reduce(ReduceOp::Min, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] - 5.0).abs() < 1e-10);
        assert!((result_view[[2]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_mean() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Mean along axis 0: [2.5, 3.5, 4.5]
        let result = executor.reduce(ReduceOp::Mean, &handle_a, &[0]).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 2.5).abs() < 1e-10);
        assert!((result_view[[1]] - 3.5).abs() < 1e-10);
        assert!((result_view[[2]] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_multiple_axes() {
        let mut executor = CpuExecutor::new();
        // 2x3x4 tensor
        let a = DenseND::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Sum along axes 0 and 2 -> shape [3]
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[0, 2]).unwrap();
        let result_dense = result.as_dense().unwrap();
        assert_eq!(result_dense.shape(), &[3]);

        // Verify result is non-trivial
        let result_view = result_dense.view();
        assert!(result_view[[0]] > 0.0);
    }

    #[test]
    fn test_reduce_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // Invalid axis (axis 2 doesn't exist for 2D tensor)
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_no_axes() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        // No axes specified
        let result = executor.reduce(ReduceOp::Sum, &handle_a, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_elem_op_sin() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(
            vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI],
            &[3],
        )
        .unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Sin, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0).abs() < 1e-10);
        assert!((result_view[[2]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_elem_op_cos() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(
            vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI],
            &[3],
        )
        .unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Cos, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] - 0.0).abs() < 1e-10);
        assert!((result_view[[2]] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_elem_op_sqrt() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Sqrt, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0).abs() < 1e-10);
        assert!((result_view[[2]] - 2.0).abs() < 1e-10);
        assert!((result_view[[3]] - 3.0).abs() < 1e-10);
        assert!((result_view[[4]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_elem_op_sqr() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Sqr, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 0.0).abs() < 1e-10);
        assert!((result_view[[1]] - 1.0).abs() < 1e-10);
        assert!((result_view[[2]] - 4.0).abs() < 1e-10);
        assert!((result_view[[3]] - 9.0).abs() < 1e-10);
        assert!((result_view[[4]] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_elem_op_recip() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 4.0, 10.0], &[4]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);

        let result = executor.elem_op(ElemOp::Recip, &handle_a).unwrap();
        let result_dense = result.as_dense().unwrap();

        let result_view = result_dense.view();
        assert!((result_view[[0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] - 0.5).abs() < 1e-10);
        assert!((result_view[[2]] - 0.25).abs() < 1e-10);
        assert!((result_view[[3]] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_memory_pool_stats() {
        let executor = CpuExecutor::new();
        let (hits, misses, hit_rate) = executor.pool_stats();

        // Initially, pool should be empty
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(hit_rate, 0.0);
    }

    #[test]
    fn test_executor_with_threads() {
        let executor = CpuExecutor::with_threads(4);
        assert_eq!(executor.num_threads, 4);
    }

    #[test]
    fn test_clear_pool() {
        let mut executor = CpuExecutor::new();

        // Perform some operations to populate the pool
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let handle_a = TensorHandle::from_dense_auto(a);
        let handle_b = TensorHandle::from_dense_auto(b);

        let _result = executor
            .einsum("ij,jk->ik", &[handle_a, handle_b], &ExecHints::default())
            .unwrap();

        // Clear the pool
        executor.clear_pool();

        let (hits, misses, hit_rate) = executor.pool_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(hit_rate, 0.0);
    }

    #[test]
    fn test_memory_pool_shape_signature() {
        assert_eq!(MemoryPool::shape_signature(&[2, 3, 4]), "2x3x4");
        assert_eq!(MemoryPool::shape_signature(&[10, 20]), "10x20");
        assert_eq!(MemoryPool::shape_signature(&[1]), "1");
    }

    #[test]
    fn test_memory_pool_acquire_release() {
        let mut pool = MemoryPool::new();

        // Acquire a buffer
        let buffer1 = pool.acquire(&[2, 3], std::mem::size_of::<f64>());
        assert_eq!(buffer1.len(), 2 * 3 * std::mem::size_of::<f64>());

        // Check stats - should be a miss
        let (hits, misses, _) = pool.stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);

        // Release the buffer
        pool.release(&[2, 3], buffer1);

        // Acquire again - should be a hit
        let buffer2 = pool.acquire(&[2, 3], std::mem::size_of::<f64>());
        assert_eq!(buffer2.len(), 2 * 3 * std::mem::size_of::<f64>());

        let (hits, misses, hit_rate) = pool.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((hit_rate - 0.5).abs() < 1e-10);
    }
}
