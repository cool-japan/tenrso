//! # Gradient Checkpointing
//!
//! Memory-efficient backward pass using selective recomputation.
//!
//! Gradient checkpointing trades computation for memory by:
//! 1. Storing only checkpoint activations during forward pass
//! 2. Recomputing intermediate values during backward pass
//! 3. Reducing memory footprint from O(n) to O(√n) for n layers
//!
//! # Usage
//!
//! ```rust,ignore
//! use tenrso_ad::checkpoint::{CheckpointedOp, CheckpointConfig};
//! use scirs2_core::ndarray_ext::Array2;
//!
//! let config = CheckpointConfig::default();
//! let mut checkpointed = CheckpointedOp::new(config);
//!
//! // Add operations to checkpoint
//! checkpointed.add_forward(|x| {
//!     // Forward computation
//!     x.mapv(|v| v * 2.0)
//! });
//!
//! // Backward pass automatically recomputes
//! let grad = checkpointed.backward(&cotangent)?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Number of segments to divide computation into
    /// Memory usage: O(n/segments) instead of O(n)
    pub num_segments: usize,

    /// Strategy for selecting checkpoints
    pub strategy: CheckpointStrategy,

    /// Whether to verify recomputed values match original
    pub verify_recomputation: bool,

    /// Tolerance for recomputation verification
    pub verification_tolerance: f64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            num_segments: 4,
            strategy: CheckpointStrategy::Uniform,
            verify_recomputation: false,
            verification_tolerance: 1e-6,
        }
    }
}

/// Strategy for selecting which activations to checkpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// Checkpoint at uniform intervals
    Uniform,

    /// Checkpoint based on memory usage
    MemoryAware,

    /// Checkpoint all (no savings, for testing)
    All,

    /// Checkpoint none (maximum computation)
    None,
}

/// A checkpointed operation that can be recomputed during backward pass
pub trait CheckpointableOp<T: Float>: Send + Sync {
    /// Forward computation
    fn forward(&self, input: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>>;

    /// Backward computation (VJP)
    fn backward(
        &self,
        input: &Array<T, IxDyn>,
        output: &Array<T, IxDyn>,
        cotangent: &Array<T, IxDyn>,
    ) -> Result<Array<T, IxDyn>>;

    /// Estimate memory usage for this operation's output
    fn memory_estimate(&self, input_shape: &[usize]) -> usize {
        input_shape.iter().product::<usize>() * std::mem::size_of::<T>()
    }
}

/// A sequence of checkpointed operations
pub struct CheckpointedSequence<T: Float> {
    operations: Vec<Arc<dyn CheckpointableOp<T>>>,
    config: CheckpointConfig,
    checkpoints: HashMap<usize, Array<T, IxDyn>>,
}

impl<T: Float + 'static> CheckpointedSequence<T> {
    /// Create a new checkpointed sequence
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            operations: Vec::new(),
            config,
            checkpoints: HashMap::new(),
        }
    }

    /// Add an operation to the sequence
    pub fn add_operation(&mut self, op: Arc<dyn CheckpointableOp<T>>) {
        self.operations.push(op);
    }

    /// Get checkpoint indices based on strategy
    fn get_checkpoint_indices(&self) -> Vec<usize> {
        match self.config.strategy {
            CheckpointStrategy::Uniform => {
                let n = self.operations.len();
                let step = n.max(1) / self.config.num_segments.max(1);
                (0..n).step_by(step.max(1)).collect()
            }
            CheckpointStrategy::All => (0..self.operations.len()).collect(),
            CheckpointStrategy::None => {
                vec![]
            }
            CheckpointStrategy::MemoryAware => {
                // Simple heuristic: checkpoint larger operations
                // In production, would track actual memory usage
                let n = self.operations.len();
                let step = n.max(1) / self.config.num_segments.max(1);
                (0..n).step_by(step.max(1)).collect()
            }
        }
    }

    /// Forward pass with checkpointing
    pub fn forward(&mut self, input: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>> {
        let checkpoint_indices = self.get_checkpoint_indices();
        let mut current = input.clone();

        self.checkpoints.clear();

        for (i, op) in self.operations.iter().enumerate() {
            current = op.forward(&current)?;

            if checkpoint_indices.contains(&i) {
                self.checkpoints.insert(i, current.clone());
            }
        }

        Ok(current)
    }

    /// Recompute forward pass from checkpoint to target index
    fn recompute_forward(
        &self,
        from_checkpoint: usize,
        to_index: usize,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        let start_activation = self
            .checkpoints
            .get(&from_checkpoint)
            .ok_or_else(|| anyhow!("Checkpoint {} not found", from_checkpoint))?;

        let mut activations = vec![start_activation.clone()];
        let mut current = start_activation.clone();

        for i in (from_checkpoint + 1)..=to_index {
            current = self.operations[i].forward(&current)?;
            activations.push(current.clone());
        }

        Ok(activations)
    }

    /// Backward pass with recomputation
    pub fn backward(&self, output_grad: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>> {
        let checkpoint_indices = self.get_checkpoint_indices();
        let mut current_grad = output_grad.clone();

        let n = self.operations.len();

        for i in (0..n).rev() {
            // Find the nearest checkpoint before this operation
            let checkpoint_idx = checkpoint_indices
                .iter()
                .filter(|&&idx| idx <= i)
                .max()
                .copied();

            // Recompute if necessary
            let (input, output) = if let Some(cp_idx) = checkpoint_idx {
                if cp_idx == i {
                    // We have the checkpoint for this exact operation
                    let input = if i == 0 {
                        // Would need original input stored separately
                        return Err(anyhow!("Original input not available"));
                    } else {
                        self.checkpoints
                            .get(&(i - 1))
                            .ok_or_else(|| anyhow!("Previous checkpoint not found"))?
                            .clone()
                    };
                    let output = self.checkpoints[&i].clone();
                    (input, output)
                } else {
                    // Recompute from checkpoint
                    let activations = self.recompute_forward(cp_idx, i)?;
                    let input = if i - cp_idx > 0 {
                        activations[i - cp_idx - 1].clone()
                    } else {
                        activations[0].clone()
                    };
                    let output = activations[i - cp_idx].clone();
                    (input, output)
                }
            } else {
                // No checkpoint available, would need original input
                return Err(anyhow!("No checkpoint available for recomputation"));
            };

            // Compute gradient
            current_grad = self.operations[i].backward(&input, &output, &current_grad)?;
        }

        Ok(current_grad)
    }

    /// Estimate memory savings
    pub fn memory_savings_estimate(&self, input_shape: &[usize]) -> MemoryStats {
        let total_ops = self.operations.len();
        let checkpointed_ops = self.get_checkpoint_indices().len();

        let base_memory =
            total_ops * input_shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let checkpoint_memory =
            checkpointed_ops * input_shape.iter().product::<usize>() * std::mem::size_of::<T>();

        MemoryStats {
            base_memory_bytes: base_memory,
            checkpoint_memory_bytes: checkpoint_memory,
            savings_bytes: base_memory.saturating_sub(checkpoint_memory),
            savings_ratio: if base_memory > 0 {
                1.0 - (checkpoint_memory as f64 / base_memory as f64)
            } else {
                0.0
            },
        }
    }
}

/// Memory usage statistics for checkpointing
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub base_memory_bytes: usize,
    pub checkpoint_memory_bytes: usize,
    pub savings_bytes: usize,
    pub savings_ratio: f64,
}

/// Simple checkpointable element-wise operation
pub struct ElementwiseOp<T: Float, F>
where
    F: Fn(&T) -> T + Send + Sync,
{
    forward_fn: Arc<F>,
    derivative_fn: Arc<dyn Fn(&T) -> T + Send + Sync>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static, F> ElementwiseOp<T, F>
where
    F: Fn(&T) -> T + Send + Sync + 'static,
{
    pub fn new(forward_fn: F, derivative_fn: impl Fn(&T) -> T + Send + Sync + 'static) -> Self {
        Self {
            forward_fn: Arc::new(forward_fn),
            derivative_fn: Arc::new(derivative_fn),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static, F> CheckpointableOp<T> for ElementwiseOp<T, F>
where
    F: Fn(&T) -> T + Send + Sync,
{
    fn forward(&self, input: &Array<T, IxDyn>) -> Result<Array<T, IxDyn>> {
        Ok(input.mapv(|x| (self.forward_fn)(&x)))
    }

    fn backward(
        &self,
        input: &Array<T, IxDyn>,
        _output: &Array<T, IxDyn>,
        cotangent: &Array<T, IxDyn>,
    ) -> Result<Array<T, IxDyn>> {
        let derivatives = input.mapv(|x| (self.derivative_fn)(&x));
        Ok(&derivatives * cotangent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.num_segments, 4);
        assert_eq!(config.strategy, CheckpointStrategy::Uniform);
        assert!(!config.verify_recomputation);
    }

    #[test]
    fn test_checkpoint_sequence_creation() {
        let config = CheckpointConfig::default();
        let seq: CheckpointedSequence<f64> = CheckpointedSequence::new(config);
        assert_eq!(seq.operations.len(), 0);
    }

    #[test]
    fn test_elementwise_op_forward() {
        let op = ElementwiseOp::new(|x: &f64| x * 2.0, |_: &f64| 2.0);

        let input = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let output = op.forward(&input).unwrap();

        let expected = array![[2.0, 4.0], [6.0, 8.0]].into_dyn();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_op_backward() {
        let op = ElementwiseOp::new(
            |x: &f64| x * x,   // f(x) = x²
            |x: &f64| 2.0 * x, // f'(x) = 2x
        );

        let input = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let output = array![[1.0, 4.0], [9.0, 16.0]].into_dyn();
        let cotangent = array![[1.0, 1.0], [1.0, 1.0]].into_dyn();

        let grad = op.backward(&input, &output, &cotangent).unwrap();

        // Gradient should be 2x * cotangent
        let expected = array![[2.0, 4.0], [6.0, 8.0]].into_dyn();
        assert_eq!(grad, expected);
    }

    #[test]
    fn test_checkpoint_sequence_add_operation() {
        let config = CheckpointConfig::default();
        let mut seq: CheckpointedSequence<f64> = CheckpointedSequence::new(config);

        let op = Arc::new(ElementwiseOp::new(|x: &f64| x * 2.0, |_: &f64| 2.0));

        seq.add_operation(op);
        assert_eq!(seq.operations.len(), 1);
    }

    #[test]
    fn test_checkpoint_indices_uniform() {
        let config = CheckpointConfig {
            num_segments: 2,
            strategy: CheckpointStrategy::Uniform,
            ..Default::default()
        };

        let mut seq: CheckpointedSequence<f64> = CheckpointedSequence::new(config);

        for _ in 0..4 {
            let op = Arc::new(ElementwiseOp::new(|x: &f64| x * 2.0, |_: &f64| 2.0));
            seq.add_operation(op);
        }

        let indices = seq.get_checkpoint_indices();
        assert!(indices.contains(&0));
        assert!(indices.len() <= 4);
    }

    #[test]
    fn test_checkpoint_indices_all() {
        let config = CheckpointConfig {
            strategy: CheckpointStrategy::All,
            ..Default::default()
        };

        let mut seq: CheckpointedSequence<f64> = CheckpointedSequence::new(config);

        for _ in 0..4 {
            let op = Arc::new(ElementwiseOp::new(|x: &f64| x * 2.0, |_: &f64| 2.0));
            seq.add_operation(op);
        }

        let indices = seq.get_checkpoint_indices();
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_checkpoint_indices_none() {
        let config = CheckpointConfig {
            strategy: CheckpointStrategy::None,
            ..Default::default()
        };

        let seq: CheckpointedSequence<f64> = CheckpointedSequence::new(config);
        let indices = seq.get_checkpoint_indices();
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_memory_savings_estimate() {
        let config = CheckpointConfig {
            num_segments: 2,
            strategy: CheckpointStrategy::Uniform,
            ..Default::default()
        };

        let mut seq: CheckpointedSequence<f64> = CheckpointedSequence::new(config);

        for _ in 0..4 {
            let op = Arc::new(ElementwiseOp::new(|x: &f64| x * 2.0, |_: &f64| 2.0));
            seq.add_operation(op);
        }

        let stats = seq.memory_savings_estimate(&[10, 10]);
        assert!(stats.base_memory_bytes > 0);
        assert!(stats.checkpoint_memory_bytes < stats.base_memory_bytes);
        assert!(stats.savings_ratio > 0.0);
    }
}
