//! # Sparse Gradient Support
//!
//! Efficient handling of sparse gradients where most components are zero.
//!
//! Sparse gradients are common in:
//! - Large networks with dropout or sparse activations
//! - Embedding layers with one-hot inputs
//! - Gradient clipping that zeros small values
//! - Distributed training with gradient compression
//!
//! # Representation
//!
//! Gradients can be stored in:
//! - **Dense format**: Full array (memory: O(n))
//! - **Sparse COO format**: (indices, values) pairs (memory: O(nnz))
//! - **Hybrid format**: Automatically switch based on sparsity
//!
//! # Usage
//!
//! ```rust,ignore
//! use tenrso_ad::sparse_grad::{SparseGradient, SparsityConfig};
//!
//! // Convert dense gradient to sparse
//! let sparse_grad = SparseGradient::from_dense(&dense_grad, threshold)?;
//!
//! // Accumulate sparse gradients efficiently
//! sparse_grad1.accumulate(&sparse_grad2)?;
//!
//! // Convert back to dense when needed
//! let dense_grad = sparse_grad.to_dense()?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;

/// Configuration for sparse gradient handling
#[derive(Debug, Clone)]
pub struct SparsityConfig {
    /// Threshold below which values are considered zero
    pub zero_threshold: f64,

    /// Target sparsity ratio (0.0 = dense, 1.0 = all zeros)
    /// Used for dynamic compression
    pub target_sparsity: f64,

    /// Automatically convert to sparse if sparsity exceeds this ratio
    pub auto_sparse_threshold: f64,

    /// Maximum number of nonzeros to store in sparse format
    /// Beyond this, use dense format
    pub max_sparse_nnz: usize,
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            zero_threshold: 1e-8,
            target_sparsity: 0.9,
            auto_sparse_threshold: 0.5,
            max_sparse_nnz: 1_000_000,
        }
    }
}

/// Sparse gradient representation
#[derive(Debug, Clone)]
pub enum SparseGradient<T: Float> {
    /// Dense representation
    Dense(Array<T, IxDyn>),

    /// Sparse COO (Coordinate) format
    /// Stores (flat_index, value) pairs
    Sparse {
        indices: Vec<usize>,
        values: Vec<T>,
        shape: Vec<usize>,
        nnz: usize,
    },
}

impl<T: Float + ScalarOperand + 'static> SparseGradient<T> {
    /// Create sparse gradient from dense array
    pub fn from_dense(dense: &Array<T, IxDyn>, threshold: T) -> Result<Self> {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val.abs() > threshold {
                indices.push(i);
                values.push(val);
            }
        }

        let nnz = indices.len();
        let shape = dense.shape().to_vec();

        Ok(Self::Sparse {
            indices,
            values,
            shape,
            nnz,
        })
    }

    /// Create from dense with automatic format selection
    pub fn from_dense_auto(dense: &Array<T, IxDyn>, config: &SparsityConfig) -> Result<Self> {
        let threshold =
            T::from(config.zero_threshold).ok_or_else(|| anyhow!("Failed to convert threshold"))?;

        let total = dense.len();
        let mut nnz = 0;

        for &val in dense.iter() {
            if val.abs() > threshold {
                nnz += 1;
            }
        }

        let sparsity = 1.0 - (nnz as f64 / total as f64);

        if sparsity >= config.auto_sparse_threshold && nnz <= config.max_sparse_nnz {
            Self::from_dense(dense, threshold)
        } else {
            Ok(Self::Dense(dense.clone()))
        }
    }

    /// Convert to dense array
    pub fn to_dense(&self) -> Result<Array<T, IxDyn>> {
        match self {
            Self::Dense(arr) => Ok(arr.clone()),
            Self::Sparse {
                indices,
                values,
                shape,
                ..
            } => {
                let mut dense = Array::zeros(IxDyn(shape));
                let flat_dense = dense
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("Non-contiguous array"))?;

                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    if idx >= flat_dense.len() {
                        return Err(anyhow!("Index {} out of bounds", idx));
                    }
                    flat_dense[idx] = val;
                }

                Ok(dense)
            }
        }
    }

    /// Get the number of nonzero elements
    pub fn nnz(&self) -> usize {
        match self {
            Self::Dense(arr) => arr.len(),
            Self::Sparse { nnz, .. } => *nnz,
        }
    }

    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        match self {
            Self::Dense(arr) => arr.len(),
            Self::Sparse { shape, .. } => shape.iter().product(),
        }
    }

    /// Get sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f64 {
        let nnz = self.nnz();
        let total = self.total_elements();

        if total == 0 {
            return 0.0;
        }

        1.0 - (nnz as f64 / total as f64)
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Dense(arr) => arr.len() * std::mem::size_of::<T>(),
            Self::Sparse {
                indices,
                values,
                shape,
                ..
            } => {
                indices.len() * std::mem::size_of::<usize>()
                    + values.len() * std::mem::size_of::<T>()
                    + shape.len() * std::mem::size_of::<usize>()
            }
        }
    }

    /// Accumulate another sparse gradient
    pub fn accumulate(&mut self, other: &Self) -> Result<()> {
        match (&mut *self, other) {
            (Self::Dense(a), Self::Dense(b)) => {
                if a.shape() != b.shape() {
                    return Err(anyhow!("Shape mismatch"));
                }
                *a = &*a + b;
                Ok(())
            }
            (Self::Sparse { .. }, Self::Sparse { .. }) => {
                // For sparse-sparse, convert to dense temporarily
                // In production, would use proper sparse addition
                let dense_self = self.to_dense()?;
                let dense_other = other.to_dense()?;
                let sum = &dense_self + &dense_other;
                *self = Self::Dense(sum);
                Ok(())
            }
            _ => {
                // Mixed sparse-dense: convert both to dense
                let dense_self = self.to_dense()?;
                let dense_other = other.to_dense()?;
                let sum = &dense_self + &dense_other;
                *self = Self::Dense(sum);
                Ok(())
            }
        }
    }

    /// Scale gradient by a scalar
    pub fn scale(&mut self, factor: T) -> Result<()> {
        match self {
            Self::Dense(arr) => {
                *arr = &*arr * factor;
                Ok(())
            }
            Self::Sparse { values, .. } => {
                for val in values.iter_mut() {
                    *val = *val * factor;
                }
                Ok(())
            }
        }
    }

    /// Compress gradient to target sparsity by zeroing small values
    pub fn compress(&mut self, config: &SparsityConfig) -> Result<()> {
        let dense = self.to_dense()?;
        let threshold = Self::compute_threshold_for_sparsity(&dense, config.target_sparsity)?;

        *self = Self::from_dense(&dense, threshold)?;
        Ok(())
    }

    /// Compute threshold that achieves target sparsity
    fn compute_threshold_for_sparsity(dense: &Array<T, IxDyn>, target_sparsity: f64) -> Result<T> {
        let mut abs_values: Vec<T> = dense.iter().map(|&x| x.abs()).collect();

        abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = abs_values.len();
        let target_nnz = ((1.0 - target_sparsity) * n as f64) as usize;

        if target_nnz >= n {
            return Ok(T::zero());
        }

        Ok(abs_values[n - target_nnz - 1])
    }
}

/// Batch operations on multiple sparse gradients
pub struct SparseGradientBatch<T: Float> {
    gradients: Vec<SparseGradient<T>>,
    config: SparsityConfig,
}

impl<T: Float + ScalarOperand + 'static> SparseGradientBatch<T> {
    pub fn new(config: SparsityConfig) -> Self {
        Self {
            gradients: Vec::new(),
            config,
        }
    }

    pub fn add(&mut self, grad: SparseGradient<T>) {
        self.gradients.push(grad);
    }

    /// Compute average of all gradients
    pub fn average(&self) -> Result<SparseGradient<T>> {
        if self.gradients.is_empty() {
            return Err(anyhow!("Empty gradient batch"));
        }

        let n = self.gradients.len();
        let mut sum = self.gradients[0].to_dense()?;

        for grad in &self.gradients[1..] {
            let dense = grad.to_dense()?;
            sum = &sum + &dense;
        }

        let divisor = T::from(n).ok_or_else(|| anyhow!("Failed to convert n"))?;
        let avg = sum / divisor;

        SparseGradient::from_dense_auto(&avg, &self.config)
    }

    /// Get total memory usage
    pub fn total_memory_bytes(&self) -> usize {
        self.gradients.iter().map(|g| g.memory_bytes()).sum()
    }

    /// Get statistics
    pub fn statistics(&self) -> GradientBatchStats {
        let total_grads = self.gradients.len();

        let sparse_count = self
            .gradients
            .iter()
            .filter(|g| matches!(g, SparseGradient::Sparse { .. }))
            .count();

        let avg_sparsity = if !self.gradients.is_empty() {
            self.gradients.iter().map(|g| g.sparsity()).sum::<f64>() / total_grads as f64
        } else {
            0.0
        };

        let total_memory = self.total_memory_bytes();

        GradientBatchStats {
            total_gradients: total_grads,
            sparse_gradients: sparse_count,
            dense_gradients: total_grads - sparse_count,
            average_sparsity: avg_sparsity,
            total_memory_bytes: total_memory,
        }
    }
}

/// Statistics for a batch of gradients
#[derive(Debug, Clone)]
pub struct GradientBatchStats {
    pub total_gradients: usize,
    pub sparse_gradients: usize,
    pub dense_gradients: usize,
    pub average_sparsity: f64,
    pub total_memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_sparsity_config_default() {
        let config = SparsityConfig::default();
        assert_eq!(config.zero_threshold, 1e-8);
        assert_eq!(config.target_sparsity, 0.9);
    }

    #[test]
    fn test_sparse_gradient_from_dense() {
        let dense = array![1.0, 0.0, 0.0, 2.0, 0.0].into_dyn();
        let threshold = 0.5;

        let sparse = SparseGradient::from_dense(&dense, threshold).unwrap();

        match sparse {
            SparseGradient::Sparse {
                indices,
                values,
                nnz,
                ..
            } => {
                assert_eq!(nnz, 2);
                assert_eq!(indices, vec![0, 3]);
                assert_eq!(values, vec![1.0, 2.0]);
            }
            _ => panic!("Expected sparse format"),
        }
    }

    #[test]
    fn test_sparse_gradient_to_dense() {
        let dense_orig = array![1.0, 0.0, 0.0, 2.0, 0.0].into_dyn();
        let threshold = 0.5;

        let sparse = SparseGradient::from_dense(&dense_orig, threshold).unwrap();
        let dense_back = sparse.to_dense().unwrap();

        let expected = array![1.0, 0.0, 0.0, 2.0, 0.0].into_dyn();
        let diff = &dense_back - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_sparse_gradient_nnz() {
        let dense = array![1.0, 0.0, 0.0, 2.0, 0.0].into_dyn();
        let threshold = 0.5;

        let sparse = SparseGradient::from_dense(&dense, threshold).unwrap();
        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_sparse_gradient_sparsity() {
        let dense = array![1.0, 0.0, 0.0, 2.0, 0.0].into_dyn();
        let threshold = 0.5;

        let sparse = SparseGradient::from_dense(&dense, threshold).unwrap();
        let sparsity = sparse.sparsity();

        // 3 zeros out of 5 elements = 60% sparsity
        assert!((sparsity - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_sparse_gradient_accumulate_dense() {
        let dense1 = array![1.0, 2.0, 3.0].into_dyn();
        let dense2 = array![4.0, 5.0, 6.0].into_dyn();

        let mut grad1 = SparseGradient::Dense(dense1);
        let grad2 = SparseGradient::Dense(dense2);

        grad1.accumulate(&grad2).unwrap();

        let result = grad1.to_dense().unwrap();
        let expected = array![5.0, 7.0, 9.0].into_dyn();

        let diff = &result - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_sparse_gradient_scale() {
        let dense = array![1.0, 2.0, 3.0].into_dyn();
        let mut grad = SparseGradient::Dense(dense);

        grad.scale(2.0).unwrap();

        let result = grad.to_dense().unwrap();
        let expected = array![2.0, 4.0, 6.0].into_dyn();

        let diff = &result - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_sparse_gradient_memory_bytes() {
        // Create a larger array to ensure sparse is actually more memory efficient
        let mut values = vec![0.0; 100];
        values[0] = 1.0;
        values[50] = 2.0;
        values[99] = 3.0;

        let dense = scirs2_core::ndarray_ext::Array::from_vec(values).into_dyn();
        let dense_grad = SparseGradient::Dense(dense.clone());

        let sparse_grad = SparseGradient::from_dense(&dense, 0.5).unwrap();

        // Sparse should use less memory (only 3 nonzeros out of 100)
        assert!(sparse_grad.memory_bytes() < dense_grad.memory_bytes());
    }

    #[test]
    fn test_sparse_gradient_batch() {
        let config = SparsityConfig::default();
        let mut batch = SparseGradientBatch::new(config);

        let grad1 = SparseGradient::Dense(array![1.0, 2.0].into_dyn());
        let grad2 = SparseGradient::Dense(array![3.0, 4.0].into_dyn());

        batch.add(grad1);
        batch.add(grad2);

        let avg = batch.average().unwrap();
        let avg_dense = avg.to_dense().unwrap();

        let expected = array![2.0, 3.0].into_dyn();
        let diff = &avg_dense - &expected;
        assert!(diff.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_sparse_gradient_batch_statistics() {
        let config = SparsityConfig::default();
        let mut batch = SparseGradientBatch::new(config);

        let grad1 = SparseGradient::Dense(array![1.0, 2.0].into_dyn());
        let grad2 = SparseGradient::Dense(array![3.0, 4.0].into_dyn());

        batch.add(grad1);
        batch.add(grad2);

        let stats = batch.statistics();
        assert_eq!(stats.total_gradients, 2);
        assert!(stats.total_memory_bytes > 0);
    }

    #[test]
    fn test_from_dense_auto_sparse() {
        let config = SparsityConfig {
            auto_sparse_threshold: 0.5,
            zero_threshold: 0.5,
            ..Default::default()
        };

        // 80% zeros = 0.8 sparsity > 0.5 threshold
        let dense = array![0.0, 0.0, 0.0, 0.0, 1.0].into_dyn();
        let grad = SparseGradient::from_dense_auto(&dense, &config).unwrap();

        assert!(matches!(grad, SparseGradient::Sparse { .. }));
    }

    #[test]
    fn test_from_dense_auto_dense() {
        let config = SparsityConfig {
            auto_sparse_threshold: 0.9,
            zero_threshold: 0.5,
            ..Default::default()
        };

        // 40% zeros = 0.4 sparsity < 0.9 threshold
        let dense = array![1.0, 2.0, 0.0, 0.0, 3.0].into_dyn();
        let grad = SparseGradient::from_dense_auto(&dense, &config).unwrap();

        assert!(matches!(grad, SparseGradient::Dense(_)));
    }
}
