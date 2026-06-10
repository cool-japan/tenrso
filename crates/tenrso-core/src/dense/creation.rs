//! Tensor creation and initialization methods
//!
//! This module provides methods for creating and initializing DenseND tensors
//! with various patterns (random, full, etc.).

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::{Num, NumCast};

#[cfg(feature = "linalg")]
use scirs2_core::ndarray_ext::{Array2, ScalarOperand};
#[cfg(feature = "linalg")]
use scirs2_core::num_traits;
#[cfg(feature = "linalg")]
use scirs2_core::numeric::{Float, NumAssign};
#[cfg(feature = "linalg")]
use std::iter::Sum;

impl<T> DenseND<T>
where
    T: Clone + Num + NumCast,
{
    /// Create a tensor with random values from a uniform distribution
    ///
    /// Uses scirs2_core::random for RNG (never rand/rand_distr directly)
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::random_uniform(&[2, 3], 0.0, 1.0);
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// // Values should be in [0.0, 1.0)
    /// ```
    pub fn random_uniform(shape: &[usize], low: f64, high: f64) -> Self
    where
        T: From<f64>,
    {
        use scirs2_core::random::quick::random_f64;
        let total: usize = shape.iter().product();
        let range = high - low;
        let data: Vec<T> = (0..total)
            .map(|_| {
                let sample: f64 = low + random_f64() * range;
                <T as From<f64>>::from(sample)
            })
            .collect();
        // `data.len() == shape.iter().product()` by construction above.
        Self {
            data: Array::from_shape_vec(IxDyn(shape), data)
                .expect("random_uniform: generated data length matches shape"),
        }
    }

    /// Create a tensor with random values from a normal distribution
    ///
    /// Uses scirs2_core::random for RNG (never rand/rand_distr directly)
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    /// * `mean` - Mean of the distribution
    /// * `std` - Standard deviation
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::random_normal(&[2, 3], 0.0, 1.0);
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
    pub fn random_normal(shape: &[usize], mean: f64, std: f64) -> Self
    where
        T: From<f64>,
    {
        use scirs2_core::random::quick::random_f64;
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total / 2 * 2)
            .step_by(2)
            .flat_map(|_| {
                let u1 = random_f64();
                let u2 = random_f64();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                vec![
                    <T as From<f64>>::from(mean + std * z0),
                    <T as From<f64>>::from(mean + std * z1),
                ]
            })
            .take(total)
            .collect();
        // `data.len() == total` by construction of the Box-Muller loop above.
        Self {
            data: Array::from_shape_vec(IxDyn(shape), data)
                .expect("random_normal: generated data length matches shape"),
        }
    }
}

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Create a tensor filled with a specific value
    ///
    /// This is an alias for `from_elem` but placed here for consistency
    /// with NumPy-style APIs.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    /// * `value` - The fill value
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::full(&[2, 3], 5.0);
    /// assert_eq!(tensor[&[0, 0]], 5.0);
    /// assert_eq!(tensor[&[1, 2]], 5.0);
    /// ```
    pub fn full(shape: &[usize], value: T) -> Self {
        Self::from_elem(shape, value)
    }

    /// Create an identity matrix (2D tensor with ones on the diagonal)
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let identity = DenseND::<f64>::eye(3);
    /// assert_eq!(identity.shape(), &[3, 3]);
    /// assert_eq!(identity[&[0, 0]], 1.0);
    /// assert_eq!(identity[&[1, 1]], 1.0);
    /// assert_eq!(identity[&[0, 1]], 0.0);
    /// ```
    pub fn eye(n: usize) -> Self {
        let mut data = Self::zeros(&[n, n]);
        for i in 0..n {
            data[&[i, i]] = T::one();
        }
        data
    }

    /// Create a 1D tensor with evenly spaced values
    ///
    /// # Arguments
    ///
    /// * `start` - Start value (inclusive)
    /// * `stop` - Stop value (exclusive)
    /// * `step` - Step size
    ///
    /// # Returns
    ///
    /// A 1D tensor with values [start, start+step, start+2*step, ...]
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::arange(0.0, 5.0, 1.0);
    /// assert_eq!(tensor.shape(), &[5]);
    /// assert_eq!(tensor[&[0]], 0.0);
    /// assert_eq!(tensor[&[4]], 4.0);
    /// ```
    pub fn arange(start: f64, stop: f64, step: f64) -> Self
    where
        T: From<f64>,
    {
        let n = ((stop - start) / step).ceil() as usize;
        let data: Vec<T> = (0..n).map(|i| T::from(start + step * i as f64)).collect();
        // `data.len() == n` by construction.
        Self::from_vec_unchecked(data, &[n])
    }

    /// Create a 1D tensor with linearly spaced values
    ///
    /// # Arguments
    ///
    /// * `start` - Start value (inclusive)
    /// * `stop` - Stop value (inclusive)
    /// * `num` - Number of values to generate
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::linspace(0.0, 10.0, 5);
    /// assert_eq!(tensor.shape(), &[5]);
    /// assert_eq!(tensor[&[0]], 0.0);
    /// assert_eq!(tensor[&[4]], 10.0);
    /// // Middle values: 2.5, 5.0, 7.5
    /// ```
    pub fn linspace(start: f64, stop: f64, num: usize) -> Self
    where
        T: From<f64>,
    {
        if num == 0 {
            // Zero-length data in a [0] shape is trivially consistent.
            return Self::from_vec_unchecked(vec![], &[0]);
        }
        if num == 1 {
            return Self::from_vec_unchecked(vec![T::from(start)], &[1]);
        }

        let step = (stop - start) / (num - 1) as f64;
        let data: Vec<T> = (0..num).map(|i| T::from(start + step * i as f64)).collect();
        // `data.len() == num` by construction.
        Self::from_vec_unchecked(data, &[num])
    }

    /// Create a one-hot encoded tensor from indices.
    ///
    /// Converts a 1D array of indices into a 2D one-hot encoded matrix.
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of class indices
    /// * `num_classes` - Total number of classes
    ///
    /// # Returns
    ///
    /// A 2D tensor of shape `[indices.len(), num_classes]` where each row is one-hot encoded.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let indices = vec![0, 2, 1, 0];
    /// let one_hot = DenseND::<f64>::one_hot(&indices, 3).unwrap();
    ///
    /// assert_eq!(one_hot.shape(), &[4, 3]);
    /// assert_eq!(one_hot[&[0, 0]], 1.0);  // First sample, class 0
    /// assert_eq!(one_hot[&[0, 1]], 0.0);
    /// assert_eq!(one_hot[&[1, 2]], 1.0);  // Second sample, class 2
    /// assert_eq!(one_hot[&[2, 1]], 1.0);  // Third sample, class 1
    /// ```
    pub fn one_hot(indices: &[usize], num_classes: usize) -> anyhow::Result<Self> {
        let num_samples = indices.len();
        let total_size = num_samples * num_classes;
        let mut data = vec![T::zero(); total_size];

        for (sample_idx, &class_idx) in indices.iter().enumerate() {
            if class_idx >= num_classes {
                anyhow::bail!(
                    "Index {} out of bounds for {} classes",
                    class_idx,
                    num_classes
                );
            }
            let flat_idx = sample_idx * num_classes + class_idx;
            data[flat_idx] = T::one();
        }

        Self::from_vec(data, &[num_samples, num_classes])
    }
}

// ============================================================
// Leverage-score factor matrix initialisation (feature = linalg)
// ============================================================

#[cfg(feature = "linalg")]
impl<T> DenseND<T>
where
    T: Clone
        + Num
        + NumCast
        + Float
        + NumAssign
        + Sum
        + Send
        + Sync
        + ScalarOperand
        + 'static,
{
    /// Creates a factor matrix for CP decomposition via leverage-score sampling.
    ///
    /// Computes the SVD of the mode-`mode` unfolding of `tensor` and returns
    /// a `[mode_size, rank]` matrix whose rows are weighted by their leverage
    /// scores (the squared-row-norms of the leading left singular vectors).
    ///
    /// High-leverage rows—those that contribute most to the low-rank structure—
    /// receive larger weights, giving a principled initialisation for CP-ALS.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Source tensor to derive leverage scores from
    /// * `mode`   - Mode along which to unfold (0-indexed)
    /// * `rank`   - Number of columns in the returned factor matrix
    ///
    /// # Returns
    ///
    /// A `DenseND<T>` of shape `[tensor.shape()[mode], rank]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `mode >= tensor.ndim()`, `rank == 0`, or
    /// `rank > tensor.shape()[mode]`.
    ///
    /// # Complexity
    ///
    /// O(mode_size × n_cols × min(mode_size, n_cols)) for the SVD.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::random_uniform(&[6, 8, 5], 0.0, 1.0);
    /// let factor = DenseND::<f64>::from_leverage_scores(&tensor, 0, 3).unwrap();
    /// assert_eq!(factor.shape(), &[6, 3]);
    /// # }
    /// ```
    #[cfg(feature = "linalg")]
    pub fn from_leverage_scores(tensor: &Self, mode: usize, rank: usize) -> anyhow::Result<Self> {
        if mode >= tensor.rank() {
            anyhow::bail!(
                "Mode {} is out of bounds for a tensor of rank {}",
                mode,
                tensor.rank()
            );
        }
        if rank == 0 {
            anyhow::bail!("Rank must be at least 1, got 0");
        }
        let mode_size = tensor.shape()[mode];
        if rank > mode_size {
            anyhow::bail!(
                "Rank {} exceeds mode-{} size {}; reduce rank or choose a larger mode",
                rank,
                mode,
                mode_size
            );
        }

        // Step 1: unfold along the requested mode → [mode_size, n_cols]
        let unfolded: Array2<T> = tensor.unfold(mode)?;

        // Step 2: full SVD — we only need U, singular values are discarded
        let (u_array, _s_array, _vt) =
            scirs2_linalg::svd(&unfolded.view(), false, None)
                .map_err(|e| anyhow::anyhow!("SVD failed in from_leverage_scores: {}", e))?;

        // u_array shape: [mode_size, min(mode_size, n_cols)]
        // Step 3: take leading `rank` columns of U
        let k = u_array.ncols().min(rank);
        let u_trunc: Array2<T> = u_array
            .slice(scirs2_core::ndarray_ext::s![.., ..k])
            .to_owned();

        // Step 4: leverage scores — squared row L2-norms of u_trunc
        let mut scores: Vec<f64> = Vec::with_capacity(mode_size);
        for row_idx in 0..mode_size {
            let row = u_trunc.row(row_idx);
            let score: f64 = row.iter().fold(0.0_f64, |acc, &v| {
                let v_f64 = <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0);
                acc + v_f64 * v_f64
            });
            scores.push(score);
        }

        // Step 5: normalise to probabilities
        let total_score: f64 = scores.iter().sum();
        let probs: Vec<f64> = if total_score > 0.0 {
            scores.iter().map(|&s| s / total_score).collect()
        } else {
            // uniform fallback when all scores are zero (degenerate tensor)
            vec![1.0 / mode_size as f64; mode_size]
        };

        // Step 6: weight each row of u_trunc by sqrt(mode_size * prob[i])
        // factor[[i, r]] = u_trunc[[i, r]] * sqrt(mode_size * prob[i])
        let mut factor_data: Vec<T> = Vec::with_capacity(mode_size * rank);
        for i in 0..mode_size {
            let sqrt_weight_f64 = (mode_size as f64 * probs[i]).sqrt();
            let sqrt_weight: T = T::from(sqrt_weight_f64)
                .ok_or_else(|| anyhow::anyhow!("cast failed: sqrt_weight for row {}", i))?;
            for r in 0..rank {
                let col = if r < k { r } else { k - 1 };
                let elem = u_trunc[(i, col)] * sqrt_weight;
                factor_data.push(elem);
            }
        }

        Self::from_vec(factor_data, &[mode_size, rank])
    }
}

// ============================================================
// Tests for from_leverage_scores (feature = linalg)
// ============================================================

#[cfg(all(test, feature = "linalg"))]
mod leverage_tests {
    use super::*;

    #[test]
    fn test_from_leverage_scores_shape() {
        let tensor = DenseND::<f64>::random_uniform(&[6, 8, 5], 0.0, 1.0);
        let factor = DenseND::<f64>::from_leverage_scores(&tensor, 0, 3).unwrap();
        assert_eq!(factor.shape(), &[6, 3]);
    }

    #[test]
    fn test_from_leverage_scores_all_modes() {
        let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        // mode 0: shape [4, rank]
        let f0 = DenseND::<f64>::from_leverage_scores(&tensor, 0, 2).unwrap();
        assert_eq!(f0.shape(), &[4, 2]);
        // mode 1: shape [5, rank]
        let f1 = DenseND::<f64>::from_leverage_scores(&tensor, 1, 3).unwrap();
        assert_eq!(f1.shape(), &[5, 3]);
        // mode 2: shape [6, rank]
        let f2 = DenseND::<f64>::from_leverage_scores(&tensor, 2, 4).unwrap();
        assert_eq!(f2.shape(), &[6, 4]);
    }

    #[test]
    fn test_from_leverage_scores_rank_1() {
        let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let factor = DenseND::<f64>::from_leverage_scores(&tensor, 1, 1).unwrap();
        assert_eq!(factor.shape(), &[5, 1]);
    }

    #[test]
    fn test_from_leverage_scores_invalid_mode() {
        let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let result = DenseND::<f64>::from_leverage_scores(&tensor, 3, 2);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("out of bounds") || msg.contains("rank"));
    }

    #[test]
    fn test_from_leverage_scores_invalid_rank() {
        let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
        let result = DenseND::<f64>::from_leverage_scores(&tensor, 0, 0);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Rank") || msg.contains("rank"));
    }

    #[test]
    fn test_from_leverage_scores_values_finite() {
        let tensor = DenseND::<f64>::random_uniform(&[5, 7, 4], 0.0, 2.0);
        let factor = DenseND::<f64>::from_leverage_scores(&tensor, 0, 3).unwrap();
        for elem in factor.iter() {
            assert!(
                elem.is_finite(),
                "Expected all elements to be finite, got {}",
                elem
            );
        }
    }
}
