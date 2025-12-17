//! Tensor creation and initialization methods
//!
//! This module provides methods for creating and initializing DenseND tensors
//! with various patterns (random, full, etc.).

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::{Num, NumCast};

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
        Self {
            data: Array::from_shape_vec(IxDyn(shape), data).unwrap(),
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
        Self {
            data: Array::from_shape_vec(IxDyn(shape), data).unwrap(),
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
        Self::from_vec(data, &[n]).unwrap()
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
            return Self::from_vec(vec![], &[0]).unwrap();
        }
        if num == 1 {
            return Self::from_vec(vec![T::from(start)], &[1]).unwrap();
        }

        let step = (stop - start) / (num - 1) as f64;
        let data: Vec<T> = (0..num).map(|i| T::from(start + step * i as f64)).collect();
        Self::from_vec(data, &[num]).unwrap()
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
