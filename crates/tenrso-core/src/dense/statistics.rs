//! Statistical operations and reductions on tensors
//!
//! This module provides comprehensive statistical functions including
//! sums, means, min/max, variance, standard deviation, and related operations.

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array, Axis, IxDyn};
use scirs2_core::numeric::{FromPrimitive, Num, NumCast};

impl<T> DenseND<T>
where
    T: Clone + Num + NumCast + FromPrimitive + std::ops::Mul<Output = T> + std::iter::Sum,
{
    /// Compute the Frobenius norm of the tensor
    ///
    /// The Frobenius norm is the square root of the sum of squares of all elements.
    /// For a tensor X, this is: ||X||_F = sqrt(Σᵢⱼₖ... X²ᵢⱼₖ...)
    /// This is equivalent to the L2 norm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[2, 3]);
    /// let norm = tensor.frobenius_norm();
    /// assert!((norm - (6.0_f64).sqrt()).abs() < 1e-10);
    /// ```
    pub fn frobenius_norm(&self) -> T
    where
        T: scirs2_core::numeric::Float,
    {
        self.data.iter().map(|&x| x * x).sum::<T>().sqrt()
    }

    /// Compute the L1 norm (Manhattan norm) of the tensor.
    ///
    /// The L1 norm is the sum of absolute values of all elements.
    /// For a tensor X, this is: ||X||_1 = Σᵢⱼₖ... |Xᵢⱼₖ...|
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[4]).unwrap();
    /// let norm = tensor.norm_l1();
    /// assert_eq!(norm, 10.0);  // |-1| + |2| + |-3| + |4| = 10
    /// ```
    pub fn norm_l1(&self) -> T
    where
        T: scirs2_core::numeric::Signed,
    {
        self.data.iter().map(|x| x.clone().abs()).sum()
    }

    /// Compute the L2 norm (Euclidean norm) of the tensor.
    ///
    /// The L2 norm is the square root of the sum of squares.
    /// This is an alias for `frobenius_norm()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![3.0, 4.0], &[2]).unwrap();
    /// let norm = tensor.norm_l2();
    /// assert_eq!(norm, 5.0);  // sqrt(3² + 4²) = 5
    /// ```
    pub fn norm_l2(&self) -> T
    where
        T: scirs2_core::numeric::Float,
    {
        self.frobenius_norm()
    }

    /// Compute the Linf norm (maximum norm) of the tensor.
    ///
    /// The Linf norm is the maximum absolute value among all elements.
    /// For a tensor X, this is: ||X||_∞ = max |Xᵢⱼₖ...|
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![-1.0, 2.0, -5.0, 3.0], &[4]).unwrap();
    /// let norm = tensor.norm_linf();
    /// assert_eq!(norm, 5.0);  // max(|-1|, |2|, |-5|, |3|) = 5
    /// ```
    pub fn norm_linf(&self) -> T
    where
        T: scirs2_core::numeric::Signed + PartialOrd,
    {
        self.data
            .iter()
            .map(|x| x.clone().abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Compute a general Lp norm of the tensor.
    ///
    /// The Lp norm is defined as: ||X||_p = (Σᵢⱼₖ... |Xᵢⱼₖ...|^p)^(1/p)
    ///
    /// # Arguments
    ///
    /// * `p` - The order of the norm (must be positive)
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// // L3 norm
    /// let norm = tensor.norm_lp(3.0);
    /// let expected = (1.0_f64.powi(3) + 2.0_f64.powi(3) + 3.0_f64.powi(3) + 4.0_f64.powi(3)).powf(1.0/3.0);
    /// assert!((norm - expected).abs() < 1e-10);
    /// ```
    pub fn norm_lp(&self, p: f64) -> T
    where
        T: scirs2_core::numeric::Float,
    {
        assert!(p > 0.0, "Norm order p must be positive");

        let p_t = NumCast::from(p).unwrap();
        let inv_p = NumCast::from(1.0 / p).unwrap();

        self.data
            .iter()
            .map(|&x| x.abs().powf(p_t))
            .sum::<T>()
            .powf(inv_p)
    }

    /// Sum all elements in the tensor.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let sum = tensor.sum();
    /// assert_eq!(sum, 21.0);
    /// ```
    pub fn sum(&self) -> T {
        self.data.iter().cloned().sum()
    }

    /// Compute the product of all elements.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let prod = tensor.prod();
    /// assert_eq!(prod, 24.0);
    /// ```
    pub fn prod(&self) -> T
    where
        T: std::iter::Product,
    {
        self.data.iter().cloned().product()
    }

    /// Compute the mean (average) of all elements.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let mean = tensor.mean();
    /// assert_eq!(mean, 3.5);
    /// ```
    pub fn mean(&self) -> T
    where
        T: std::ops::Div<Output = T>,
    {
        let sum = self.sum();
        let count = NumCast::from(self.len()).unwrap();
        sum / count
    }

    /// Sum elements along a specific axis.
    ///
    /// Returns a tensor with one fewer dimension (or same shape with size-1 dimension if keepdims=true),
    /// where the specified axis has been summed out.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to sum
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// // Sum along axis 0 (collapse rows)
    /// let sum_axis0 = tensor.sum_axis(0, false).unwrap();
    /// assert_eq!(sum_axis0.shape(), &[3]);
    ///
    /// // Sum with keepdims
    /// let sum_keepdims = tensor.sum_axis(0, true).unwrap();
    /// assert_eq!(sum_keepdims.shape(), &[1, 3]);
    /// ```
    pub fn sum_axis(&self, axis: usize, keepdims: bool) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }
        let summed = self.data.sum_axis(Axis(axis));
        let mut result = Self { data: summed };
        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }

    /// Compute the product of elements along a specific axis.
    ///
    /// Returns a tensor with one fewer dimension (or same shape with size-1 dimension if keepdims=true),
    /// where the specified axis has been reduced by multiplication.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the product
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// // Product along axis 1
    /// let prod_axis1 = tensor.prod_axis(1, false).unwrap();
    /// assert_eq!(prod_axis1.shape(), &[2]);
    /// assert_eq!(prod_axis1[&[0]], 6.0);  // 1*2*3
    /// assert_eq!(prod_axis1[&[1]], 120.0); // 4*5*6
    ///
    /// // Product with keepdims
    /// let prod_keepdims = tensor.prod_axis(1, true).unwrap();
    /// assert_eq!(prod_keepdims.shape(), &[2, 1]);
    /// ```
    pub fn prod_axis(&self, axis: usize, keepdims: bool) -> anyhow::Result<Self>
    where
        T: std::iter::Product,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // Use ndarray's product operation along axis
        let product_data = {
            // Calculate the output shape
            let mut output_shape: Vec<usize> = self.shape().to_vec();
            output_shape.remove(axis);

            if output_shape.is_empty() {
                // Scalar result
                return Ok(Self::from_elem(&[1], self.prod()));
            }

            let output_size: usize = output_shape.iter().product();
            let axis_size = self.shape()[axis];
            let mut result_data = Vec::with_capacity(output_size);

            // Compute product for each position
            for i in 0..output_size {
                let mut product = T::one();

                for j in 0..axis_size {
                    // Convert flat index to multi-dimensional index
                    let mut indices = Vec::with_capacity(self.rank());
                    let mut remaining = i;
                    let mut axis_inserted = false;

                    for (dim_idx, _) in self.shape().iter().enumerate() {
                        if dim_idx == axis {
                            indices.push(j);
                            axis_inserted = true;
                        } else {
                            let actual_idx = if !axis_inserted && dim_idx < axis {
                                dim_idx
                            } else if axis_inserted && dim_idx > axis {
                                dim_idx - 1
                            } else {
                                dim_idx
                            };

                            let stride: usize = output_shape[actual_idx + 1..]
                                .iter()
                                .product::<usize>()
                                .max(1);
                            indices.push(remaining / stride);
                            remaining %= stride;
                        }
                    }

                    product = product * self.data[&indices[..]].clone();
                }

                result_data.push(product);
            }

            result_data
        };

        let mut output_shape: Vec<usize> = self.shape().to_vec();
        output_shape.remove(axis);
        let mut result = Self::from_vec(product_data, &output_shape)?;
        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }

    /// Compute mean along a specific axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the mean
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let mean_axis0 = tensor.mean_axis(0, false).unwrap();
    /// assert_eq!(mean_axis0.shape(), &[3]);
    ///
    /// let mean_keepdims = tensor.mean_axis(0, true).unwrap();
    /// assert_eq!(mean_keepdims.shape(), &[1, 3]);
    /// ```
    pub fn mean_axis(&self, axis: usize, keepdims: bool) -> anyhow::Result<Self>
    where
        T: std::ops::Div<Output = T>,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }
        let mean_arr = self
            .data
            .mean_axis(Axis(axis))
            .ok_or_else(|| anyhow::anyhow!("Failed to compute mean along axis {}", axis))?;
        let mut result = Self { data: mean_arr };
        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }
}

impl<T> DenseND<T>
where
    T: Clone + Num + PartialOrd,
{
    /// Find the minimum element in the tensor.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let min = tensor.min();
    /// assert_eq!(min, &1.0);
    /// ```
    pub fn min(&self) -> &T {
        self.data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Find the maximum element in the tensor.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let max = tensor.max();
    /// assert_eq!(max, &9.0);
    /// ```
    pub fn max(&self) -> &T {
        self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Find the index of the maximum element (flattened)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0], &[5]).unwrap();
    /// assert_eq!(tensor.argmax(), 3);
    /// ```
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Find the index of the minimum element (flattened)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![3.0, 1.0, 2.0, 5.0, 0.5], &[5]).unwrap();
    /// assert_eq!(tensor.argmin(), 4);
    /// ```
    pub fn argmin(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }
}

impl<T> DenseND<T>
where
    T: Clone
        + Num
        + NumCast
        + FromPrimitive
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum,
{
    /// Compute variance of all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let var = tensor.variance();
    /// assert!((var - 2.0).abs() < 1e-10);
    /// ```
    pub fn variance(&self) -> T {
        let mean_val = self.mean();
        let squared_diffs: T = self
            .data
            .iter()
            .map(|x| {
                let diff = x.clone() - mean_val.clone();
                diff.clone() * diff
            })
            .sum();
        let n = NumCast::from(self.len()).unwrap();
        squared_diffs / n
    }

    /// Compute standard deviation of all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let std = tensor.std();
    /// assert!((std - 2.0_f64.sqrt()).abs() < 1e-10);
    /// ```
    pub fn std(&self) -> T
    where
        T: scirs2_core::numeric::Float,
    {
        self.variance().sqrt()
    }

    /// Compute variance along a specific axis
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute variance
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let var_axis0 = tensor.variance_axis(0, false).unwrap();
    /// assert_eq!(var_axis0.shape(), &[3]);
    ///
    /// let var_keepdims = tensor.variance_axis(0, true).unwrap();
    /// assert_eq!(var_keepdims.shape(), &[1, 3]);
    /// ```
    pub fn variance_axis(&self, axis: usize, keepdims: bool) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // Compute mean with keepdims=true for broadcasting
        let mean = self.mean_axis(axis, true)?;
        let mean_broadcasted = mean
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast mean to original shape"))?;

        let squared_diffs: Array<T, IxDyn> = self
            .data
            .iter()
            .zip(mean_broadcasted.iter())
            .map(|(x, m)| {
                let diff = x.clone() - m.clone();
                diff.clone() * diff
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Array<T, _>>()
            .into_shape_with_order(IxDyn(self.shape()))
            .unwrap();

        let variance_data = squared_diffs
            .mean_axis(Axis(axis))
            .ok_or_else(|| anyhow::anyhow!("Failed to compute variance along axis {}", axis))?;

        let mut result = Self {
            data: variance_data,
        };

        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }

    /// Compute the cumulative sum of elements along an axis.
    ///
    /// Returns a tensor of the same shape where each element is the cumulative
    /// sum up to that position along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute cumulative sums
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let cumsum = tensor.cumsum(1).unwrap();
    /// // Row 0: [1, 3, 6]  (1, 1+2, 1+2+3)
    /// // Row 1: [4, 9, 15] (4, 4+5, 4+5+6)
    /// assert_eq!(cumsum[&[0, 0]], 1.0);
    /// assert_eq!(cumsum[&[0, 1]], 3.0);
    /// assert_eq!(cumsum[&[0, 2]], 6.0);
    /// ```
    pub fn cumsum(&self, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let mut result = self.clone();
        let shape = self.shape().to_vec();
        let axis_size = shape[axis];

        // Compute cumulative sum along the axis
        for slice_idx in 0..self.len() / axis_size {
            for pos in 1..axis_size {
                // Build indices for current and previous positions
                let mut curr_indices = Vec::with_capacity(self.rank());
                let mut prev_indices = Vec::with_capacity(self.rank());
                let mut remaining = slice_idx;

                for (dim_idx, _) in shape.iter().enumerate() {
                    if dim_idx == axis {
                        curr_indices.push(pos);
                        prev_indices.push(pos - 1);
                    } else {
                        let other_dims: usize = shape
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != axis && *i > dim_idx)
                            .map(|(_, &s)| s)
                            .product::<usize>()
                            .max(1);
                        let idx = remaining / other_dims;
                        remaining %= other_dims;
                        curr_indices.push(idx);
                        prev_indices.push(idx);
                    }
                }

                let prev_val = result.data[&prev_indices[..]].clone();
                let curr_val = self.data[&curr_indices[..]].clone();
                result.data[&curr_indices[..]] = prev_val + curr_val;
            }
        }

        Ok(result)
    }

    /// Compute the cumulative product of elements along an axis.
    ///
    /// Returns a tensor of the same shape where each element is the cumulative
    /// product up to that position along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute cumulative products
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 2.0, 2.0, 2.0], &[2, 3]).unwrap();
    /// let cumprod = tensor.cumprod(1).unwrap();
    /// // Row 0: [1, 2, 6]   (1, 1*2, 1*2*3)
    /// // Row 1: [2, 4, 8]   (2, 2*2, 2*2*2)
    /// assert_eq!(cumprod[&[0, 0]], 1.0);
    /// assert_eq!(cumprod[&[0, 2]], 6.0);
    /// assert_eq!(cumprod[&[1, 2]], 8.0);
    /// ```
    pub fn cumprod(&self, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let mut result = self.clone();
        let shape = self.shape().to_vec();
        let axis_size = shape[axis];

        // Compute cumulative product along the axis
        for slice_idx in 0..self.len() / axis_size {
            for pos in 1..axis_size {
                // Build indices for current and previous positions
                let mut curr_indices = Vec::with_capacity(self.rank());
                let mut prev_indices = Vec::with_capacity(self.rank());
                let mut remaining = slice_idx;

                for (dim_idx, _) in shape.iter().enumerate() {
                    if dim_idx == axis {
                        curr_indices.push(pos);
                        prev_indices.push(pos - 1);
                    } else {
                        let other_dims: usize = shape
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != axis && *i > dim_idx)
                            .map(|(_, &s)| s)
                            .product::<usize>()
                            .max(1);
                        let idx = remaining / other_dims;
                        remaining %= other_dims;
                        curr_indices.push(idx);
                        prev_indices.push(idx);
                    }
                }

                let prev_val = result.data[&prev_indices[..]].clone();
                let curr_val = self.data[&curr_indices[..]].clone();
                result.data[&curr_indices[..]] = prev_val * curr_val;
            }
        }

        Ok(result)
    }

    /// Compute the median of all elements
    ///
    /// # Complexity
    ///
    /// O(n log n) where n is the number of elements (requires sorting)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();
    /// let median = tensor.median();
    /// assert_eq!(median, 3.0);
    /// ```
    pub fn median(&self) -> T
    where
        T: PartialOrd,
    {
        let mut sorted: Vec<T> = self.data.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2].clone()
        } else {
            let mid1 = sorted[n / 2 - 1].clone();
            let mid2 = sorted[n / 2].clone();
            let two = NumCast::from(2).unwrap();
            (mid1 + mid2) / two
        }
    }

    /// Compute the quantile at a given probability
    ///
    /// # Arguments
    ///
    /// * `q` - The quantile probability (must be in range [0, 1])
    ///
    /// # Complexity
    ///
    /// O(n log n) where n is the number of elements (requires sorting)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let q25 = tensor.quantile(0.25);
    /// assert_eq!(q25, 2.0);
    /// ```
    pub fn quantile(&self, q: f64) -> T
    where
        T: PartialOrd + scirs2_core::numeric::Float,
    {
        assert!((0.0..=1.0).contains(&q), "Quantile must be in range [0, 1]");

        let mut sorted: Vec<T> = self.data.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }

        // Linear interpolation method
        let index = q * ((n - 1) as f64);
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        if lower_idx == upper_idx {
            sorted[lower_idx]
        } else {
            let lower_val = sorted[lower_idx];
            let upper_val = sorted[upper_idx];
            let weight = NumCast::from(index - lower_idx as f64).unwrap();
            lower_val + (upper_val - lower_val) * weight
        }
    }

    /// Compute the percentile at a given percent
    ///
    /// This is a convenience wrapper around `quantile` that takes a percentage (0-100)
    /// instead of a probability (0-1).
    ///
    /// # Arguments
    ///
    /// * `p` - The percentile (must be in range [0, 100])
    ///
    /// # Complexity
    ///
    /// O(n log n) where n is the number of elements (requires sorting)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let p75 = tensor.percentile(75.0);
    /// assert_eq!(p75, 4.0);
    /// ```
    pub fn percentile(&self, p: f64) -> T
    where
        T: PartialOrd + scirs2_core::numeric::Float,
    {
        assert!(
            (0.0..=100.0).contains(&p),
            "Percentile must be in range [0, 100]"
        );
        self.quantile(p / 100.0)
    }

    /// Compute the median along a specific axis
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the median
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Complexity
    ///
    /// O(n log m) where n is the number of elements and m is the size along the axis
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let median_axis0 = tensor.median_axis(0, false).unwrap();
    /// assert_eq!(median_axis0.shape(), &[3]);
    ///
    /// let median_keepdims = tensor.median_axis(0, true).unwrap();
    /// assert_eq!(median_keepdims.shape(), &[1, 3]);
    /// ```
    pub fn median_axis(&self, axis: usize, keepdims: bool) -> anyhow::Result<Self>
    where
        T: PartialOrd,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // Calculate output shape
        let mut output_shape: Vec<usize> = self.shape().to_vec();
        output_shape.remove(axis);

        if output_shape.is_empty() {
            // If removing the axis leaves us with a scalar
            return Ok(Self::from_elem(&[1], self.median()));
        }

        let output_size: usize = output_shape.iter().product();
        let axis_size = self.shape()[axis];

        // Collect values along the axis and compute median for each slice
        let mut result_data = Vec::with_capacity(output_size);

        // For each position in the output
        for i in 0..output_size {
            let mut values = Vec::with_capacity(axis_size);

            // Collect values along the axis
            for j in 0..axis_size {
                // Convert flat index to multi-dimensional index
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = i;
                let mut axis_inserted = false;

                for (dim_idx, &_dim_size) in self.shape().iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(j);
                        axis_inserted = true;
                    } else {
                        let actual_idx = if !axis_inserted && dim_idx < axis {
                            dim_idx
                        } else if axis_inserted && dim_idx > axis {
                            dim_idx - 1
                        } else {
                            dim_idx
                        };

                        let stride: usize = output_shape[actual_idx + 1..]
                            .iter()
                            .product::<usize>()
                            .max(1);
                        indices.push(remaining / stride);
                        remaining %= stride;
                    }
                }

                values.push(self.data[&indices[..]].clone());
            }

            // Compute median of collected values
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if axis_size % 2 == 1 {
                values[axis_size / 2].clone()
            } else {
                let mid1 = values[axis_size / 2 - 1].clone();
                let mid2 = values[axis_size / 2].clone();
                let two = NumCast::from(2).unwrap();
                (mid1 + mid2) / two
            };

            result_data.push(median);
        }

        let mut result = Self::from_vec(result_data, &output_shape)?;
        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }

    /// Compute the quantile along a specific axis
    ///
    /// # Arguments
    ///
    /// * `q` - The quantile probability (must be in range [0, 1])
    /// * `axis` - The axis along which to compute the quantile
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Complexity
    ///
    /// O(n log m) where n is the number of elements and m is the size along the axis
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let q75_axis0 = tensor.quantile_axis(0.75, 0, false).unwrap();
    /// assert_eq!(q75_axis0.shape(), &[3]);
    ///
    /// let q75_keepdims = tensor.quantile_axis(0.75, 0, true).unwrap();
    /// assert_eq!(q75_keepdims.shape(), &[1, 3]);
    /// ```
    pub fn quantile_axis(&self, q: f64, axis: usize, keepdims: bool) -> anyhow::Result<Self>
    where
        T: PartialOrd + scirs2_core::numeric::Float,
    {
        assert!((0.0..=1.0).contains(&q), "Quantile must be in range [0, 1]");

        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // Calculate output shape
        let mut output_shape: Vec<usize> = self.shape().to_vec();
        output_shape.remove(axis);

        if output_shape.is_empty() {
            // If removing the axis leaves us with a scalar
            return Ok(Self::from_elem(&[1], self.quantile(q)));
        }

        let output_size: usize = output_shape.iter().product();
        let axis_size = self.shape()[axis];

        // Collect values along the axis and compute quantile for each slice
        let mut result_data = Vec::with_capacity(output_size);

        // For each position in the output
        for i in 0..output_size {
            let mut values = Vec::with_capacity(axis_size);

            // Collect values along the axis
            for j in 0..axis_size {
                // Convert flat index to multi-dimensional index
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = i;
                let mut axis_inserted = false;

                for (dim_idx, &_dim_size) in self.shape().iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(j);
                        axis_inserted = true;
                    } else {
                        let actual_idx = if !axis_inserted && dim_idx < axis {
                            dim_idx
                        } else if axis_inserted && dim_idx > axis {
                            dim_idx - 1
                        } else {
                            dim_idx
                        };

                        let stride: usize = output_shape[actual_idx + 1..]
                            .iter()
                            .product::<usize>()
                            .max(1);
                        indices.push(remaining / stride);
                        remaining %= stride;
                    }
                }

                values.push(self.data[&indices[..]]);
            }

            // Compute quantile of collected values
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let quantile = if values.len() == 1 {
                values[0]
            } else {
                let index = q * ((values.len() - 1) as f64);
                let lower_idx = index.floor() as usize;
                let upper_idx = index.ceil() as usize;

                if lower_idx == upper_idx {
                    values[lower_idx]
                } else {
                    let lower_val = values[lower_idx];
                    let upper_val = values[upper_idx];
                    let weight = NumCast::from(index - lower_idx as f64).unwrap();
                    lower_val + (upper_val - lower_val) * weight
                }
            };

            result_data.push(quantile);
        }

        let mut result = Self::from_vec(result_data, &output_shape)?;
        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }

    /// Compute the percentile along a specific axis
    ///
    /// This is a convenience wrapper around `quantile_axis` that takes a percentage (0-100)
    /// instead of a probability (0-1).
    ///
    /// # Arguments
    ///
    /// * `p` - The percentile (must be in range [0, 100])
    /// * `axis` - The axis along which to compute the percentile
    /// * `keepdims` - If true, retains the reduced dimension with size 1
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let p50_axis1 = tensor.percentile_axis(50.0, 1, false).unwrap();
    /// assert_eq!(p50_axis1.shape(), &[2]);
    ///
    /// let p50_keepdims = tensor.percentile_axis(50.0, 1, true).unwrap();
    /// assert_eq!(p50_keepdims.shape(), &[2, 1]);
    /// ```
    pub fn percentile_axis(&self, p: f64, axis: usize, keepdims: bool) -> anyhow::Result<Self>
    where
        T: PartialOrd + scirs2_core::numeric::Float,
    {
        assert!(
            (0.0..=100.0).contains(&p),
            "Percentile must be in range [0, 100]"
        );
        self.quantile_axis(p / 100.0, axis, keepdims)
    }

    /// Compute the covariance between two 1D tensors
    ///
    /// The covariance measures how much two variables change together:
    /// cov(X, Y) = E\[(X - E\[X\])(Y - E\[Y\])\]
    ///
    /// This is the unbiased estimator (divides by n-1).
    ///
    /// # Arguments
    ///
    /// * `other` - The other 1D tensor to compute covariance with
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either tensor is not 1D
    /// - The tensors have different lengths
    /// - Either tensor has fewer than 2 elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let x = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let y = DenseND::<f64>::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
    ///
    /// let cov = x.covariance(&y).unwrap();
    /// assert!((cov - 5.0).abs() < 1e-10);  // Perfect positive correlation
    /// ```
    pub fn covariance(&self, other: &Self) -> anyhow::Result<T>
    where
        T: scirs2_core::numeric::Float,
    {
        anyhow::ensure!(
            self.rank() == 1,
            "covariance requires 1D tensors, got rank {}",
            self.rank()
        );
        anyhow::ensure!(
            other.rank() == 1,
            "covariance requires 1D tensors, got rank {}",
            other.rank()
        );
        anyhow::ensure!(
            self.shape() == other.shape(),
            "Tensors must have same shape for covariance"
        );

        let n = self.shape()[0];
        anyhow::ensure!(n >= 2, "Need at least 2 elements for covariance, got {}", n);

        let mean_x = self.mean();
        let mean_y = other.mean();

        let n_minus_1 = T::from_usize(n - 1).unwrap();

        let cov = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<T>()
            / n_minus_1;

        Ok(cov)
    }

    /// Compute the covariance matrix for a 2D tensor
    ///
    /// For a tensor of shape (n_samples, n_features), computes the n_features × n_features
    /// covariance matrix where element (i, j) is the covariance between features i and j.
    ///
    /// Uses unbiased estimator (divides by n-1).
    ///
    /// # Arguments
    ///
    /// * `rowvar` - If true, each row is a variable (default). If false, each column is a variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D or has fewer than 2 samples
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // 3 samples, 2 features
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[3, 2]
    /// ).unwrap();
    ///
    /// let cov_matrix = data.covariance_matrix(false).unwrap();
    /// assert_eq!(cov_matrix.shape(), &[2, 2]);
    /// ```
    pub fn covariance_matrix(&self, rowvar: bool) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        anyhow::ensure!(
            self.rank() == 2,
            "covariance_matrix requires 2D tensor, got rank {}",
            self.rank()
        );

        // Transpose if rowvar is false (so rows are features)
        let working = if rowvar {
            self.clone()
        } else {
            self.transpose()?
        };

        let n_features = working.shape()[0];
        let n_samples = working.shape()[1];

        anyhow::ensure!(
            n_samples >= 2,
            "Need at least 2 samples for covariance, got {}",
            n_samples
        );

        // Compute mean for each feature (row)
        let means = working.mean_axis(1, false)?;

        // Center the data
        let mut centered_data = Vec::with_capacity(n_features * n_samples);
        for i in 0..n_features {
            let mean_i = means[&[i]];
            for j in 0..n_samples {
                let val = working[&[i, j]];
                centered_data.push(val - mean_i);
            }
        }
        let centered = Self::from_vec(centered_data, &[n_features, n_samples])?;

        // Compute covariance matrix: C = (1/(n-1)) * X * X^T
        let n_minus_1 = T::from_usize(n_samples - 1).unwrap();

        let mut cov_data = Vec::with_capacity(n_features * n_features);
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = T::zero();
                for k in 0..n_samples {
                    sum = sum + centered[&[i, k]] * centered[&[j, k]];
                }
                cov_data.push(sum / n_minus_1);
            }
        }

        Self::from_vec(cov_data, &[n_features, n_features])
    }

    /// Compute the Pearson correlation coefficient between two 1D tensors
    ///
    /// The correlation coefficient measures the linear relationship between two variables,
    /// normalized to the range [-1, 1]:
    /// corr(X, Y) = cov(X, Y) / (std(X) * std(Y))
    ///
    /// # Arguments
    ///
    /// * `other` - The other 1D tensor to correlate with
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either tensor is not 1D
    /// - The tensors have different lengths
    /// - Either tensor has zero variance
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let x = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let y = DenseND::<f64>::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
    ///
    /// let corr = x.correlation(&y).unwrap();
    /// assert!((corr - 1.0).abs() < 1e-10);  // Perfect positive correlation
    /// ```
    pub fn correlation(&self, other: &Self) -> anyhow::Result<T>
    where
        T: scirs2_core::numeric::Float,
    {
        anyhow::ensure!(
            self.rank() == 1,
            "correlation requires 1D tensors, got rank {}",
            self.rank()
        );
        anyhow::ensure!(
            other.rank() == 1,
            "correlation requires 1D tensors, got rank {}",
            other.rank()
        );
        anyhow::ensure!(
            self.shape() == other.shape(),
            "Tensors must have same shape for correlation"
        );

        let n = self.shape()[0];
        anyhow::ensure!(
            n >= 2,
            "Need at least 2 elements for correlation, got {}",
            n
        );

        let mean_x = self.mean();
        let mean_y = other.mean();

        // Compute covariance and variances in one pass for numerical stability
        let mut cov_xy = T::zero();
        let mut var_x = T::zero();
        let mut var_y = T::zero();

        for (&x, &y) in self.data.iter().zip(other.data.iter()) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov_xy = cov_xy + dx * dy;
            var_x = var_x + dx * dx;
            var_y = var_y + dy * dy;
        }

        anyhow::ensure!(
            var_x > T::zero() && var_y > T::zero(),
            "Cannot compute correlation with zero variance"
        );

        // Correlation coefficient: cov / (std_x * std_y) = cov_xy / sqrt(var_x * var_y)
        Ok(cov_xy / (var_x * var_y).sqrt())
    }

    /// Compute the correlation matrix for a 2D tensor
    ///
    /// For a tensor of shape (n_samples, n_features), computes the n_features × n_features
    /// correlation matrix where element (i, j) is the Pearson correlation between features i and j.
    ///
    /// The diagonal elements are always 1.0 (perfect self-correlation).
    ///
    /// # Arguments
    ///
    /// * `rowvar` - If true, each row is a variable. If false, each column is a variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D or any feature has zero variance
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // 5 samples, 2 features
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
    ///     &[5, 2]
    /// ).unwrap();
    ///
    /// let corr_matrix = data.correlation_matrix(false).unwrap();
    /// assert_eq!(corr_matrix.shape(), &[2, 2]);
    /// // Diagonal should be 1.0
    /// assert!((corr_matrix[&[0, 0]] - 1.0).abs() < 1e-10);
    /// assert!((corr_matrix[&[1, 1]] - 1.0).abs() < 1e-10);
    /// ```
    pub fn correlation_matrix(&self, rowvar: bool) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        let cov_matrix = self.covariance_matrix(rowvar)?;
        let n = cov_matrix.shape()[0];

        // Extract standard deviations (sqrt of diagonal elements)
        let mut stds = Vec::with_capacity(n);
        for i in 0..n {
            let variance = cov_matrix[&[i, i]];
            anyhow::ensure!(
                variance > T::zero(),
                "Cannot compute correlation with zero variance in feature {}",
                i
            );
            stds.push(variance.sqrt());
        }

        // Normalize covariance by standard deviations
        let mut corr_data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let cov = cov_matrix[&[i, j]];
                let corr = cov / (stds[i] * stds[j]);
                corr_data.push(corr);
            }
        }

        Self::from_vec(corr_data, &[n, n])
    }

    /// Batch normalization: normalize along the batch dimension (axis 0).
    ///
    /// Normalizes the input by subtracting the mean and dividing by the standard deviation
    /// (plus epsilon for numerical stability), then scales and shifts by learned parameters.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small constant for numerical stability (typically 1e-5)
    /// * `gamma` - Optional scale parameter (defaults to 1.0 if None)
    /// * `beta` - Optional shift parameter (defaults to 0.0 if None)
    ///
    /// # Returns
    ///
    /// Normalized tensor with the same shape
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[3, 2]
    /// ).unwrap();
    ///
    /// let normalized = data.batch_norm(1e-5, None, None).unwrap();
    /// assert_eq!(normalized.shape(), &[3, 2]);
    ///
    /// // With scale and shift
    /// let gamma = DenseND::from_vec(vec![2.0, 2.0], &[2]).unwrap();
    /// let beta = DenseND::from_vec(vec![0.5, 0.5], &[2]).unwrap();
    /// let normalized = data.batch_norm(1e-5, Some(&gamma), Some(&beta)).unwrap();
    /// ```
    pub fn batch_norm(
        &self,
        epsilon: T,
        gamma: Option<&Self>,
        beta: Option<&Self>,
    ) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        if self.rank() < 2 {
            anyhow::bail!(
                "batch_norm requires at least 2D tensor, got rank {}",
                self.rank()
            );
        }

        // Compute mean and variance along batch dimension (axis 0)
        let mean = self.mean_axis(0, true)?;
        let var = self.variance_axis(0, true)?;

        // Normalize: (x - mean) / sqrt(var + epsilon)
        let mean_broadcast = mean
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast mean"))?;
        let var_broadcast = var
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast variance"))?;

        let mut normalized_data = Vec::with_capacity(self.len());
        for ((&x, &m), &v) in self
            .data
            .iter()
            .zip(mean_broadcast.iter())
            .zip(var_broadcast.iter())
        {
            let normalized = (x - m) / (v + epsilon).sqrt();
            normalized_data.push(normalized);
        }

        let mut result = Self::from_vec(normalized_data, self.shape())?;

        // Apply scale (gamma) if provided
        if let Some(gamma_tensor) = gamma {
            if gamma_tensor.shape() != &self.shape()[1..] {
                anyhow::bail!(
                    "gamma shape {:?} does not match feature shape {:?}",
                    gamma_tensor.shape(),
                    &self.shape()[1..]
                );
            }
            let gamma_broadcast = gamma_tensor
                .data
                .broadcast(self.shape())
                .ok_or_else(|| anyhow::anyhow!("Failed to broadcast gamma"))?;
            let scaled_data: Vec<T> = result
                .data
                .iter()
                .zip(gamma_broadcast.iter())
                .map(|(&x, &g)| x * g)
                .collect();
            result = Self::from_vec(scaled_data, self.shape())?;
        }

        // Apply shift (beta) if provided
        if let Some(beta_tensor) = beta {
            if beta_tensor.shape() != &self.shape()[1..] {
                anyhow::bail!(
                    "beta shape {:?} does not match feature shape {:?}",
                    beta_tensor.shape(),
                    &self.shape()[1..]
                );
            }
            let beta_broadcast = beta_tensor
                .data
                .broadcast(self.shape())
                .ok_or_else(|| anyhow::anyhow!("Failed to broadcast beta"))?;
            let shifted_data: Vec<T> = result
                .data
                .iter()
                .zip(beta_broadcast.iter())
                .map(|(&x, &b)| x + b)
                .collect();
            result = Self::from_vec(shifted_data, self.shape())?;
        }

        Ok(result)
    }

    /// Layer normalization: normalize along the last dimension(s).
    ///
    /// Normalizes the input by subtracting the mean and dividing by the standard deviation
    /// across the specified normalized shape, then scales and shifts by learned parameters.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The shape over which to normalize (typically the last dimension(s))
    /// * `epsilon` - Small constant for numerical stability (typically 1e-5)
    /// * `gamma` - Optional scale parameter
    /// * `beta` - Optional shift parameter
    ///
    /// # Returns
    ///
    /// Normalized tensor with the same shape
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// // Normalize over the last dimension (features)
    /// let normalized = data.layer_norm(&[3], 1e-5, None, None).unwrap();
    /// assert_eq!(normalized.shape(), &[2, 3]);
    /// ```
    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        epsilon: T,
        gamma: Option<&Self>,
        beta: Option<&Self>,
    ) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        // Verify normalized_shape matches the last dimensions of self
        let rank = self.rank();
        let norm_rank = normalized_shape.len();
        if norm_rank > rank {
            anyhow::bail!(
                "normalized_shape has {} dimensions but tensor has only {} dimensions",
                norm_rank,
                rank
            );
        }

        let start_axis = rank - norm_rank;
        if self.shape()[start_axis..] != *normalized_shape {
            anyhow::bail!(
                "normalized_shape {:?} does not match tensor's last {} dimensions {:?}",
                normalized_shape,
                norm_rank,
                &self.shape()[start_axis..]
            );
        }

        // Calculate batch_shape and feature_shape
        let batch_shape = &self.shape()[..start_axis];
        let batch_size: usize = batch_shape.iter().product();
        let feature_size: usize = normalized_shape.iter().product();

        // Reshape to [batch_size, feature_size]
        let flat = self.reshape(&[batch_size, feature_size])?;

        // Compute mean and std for each batch element
        let mut normalized_data = Vec::with_capacity(flat.len());
        for i in 0..batch_size {
            let mut sum = T::zero();
            let mut sq_sum = T::zero();

            for j in 0..feature_size {
                let val = flat[&[i, j]];
                sum = sum + val;
                sq_sum = sq_sum + val * val;
            }

            let mean = sum / T::from_usize(feature_size).unwrap();
            let variance = sq_sum / T::from_usize(feature_size).unwrap() - mean * mean;
            let std = (variance + epsilon).sqrt();

            for j in 0..feature_size {
                let val = flat[&[i, j]];
                let normalized = (val - mean) / std;
                normalized_data.push(normalized);
            }
        }

        let mut result = Self::from_vec(normalized_data, self.shape())?;

        // Apply scale (gamma) and shift (beta) if provided
        if let Some(gamma_tensor) = gamma {
            if gamma_tensor.shape() != normalized_shape {
                anyhow::bail!(
                    "gamma shape {:?} does not match normalized_shape {:?}",
                    gamma_tensor.shape(),
                    normalized_shape
                );
            }
            let gamma_broadcast = gamma_tensor
                .data
                .broadcast(self.shape())
                .ok_or_else(|| anyhow::anyhow!("Failed to broadcast gamma"))?;
            let scaled_data: Vec<T> = result
                .data
                .iter()
                .zip(gamma_broadcast.iter())
                .map(|(&x, &g)| x * g)
                .collect();
            result = Self::from_vec(scaled_data, self.shape())?;
        }

        if let Some(beta_tensor) = beta {
            if beta_tensor.shape() != normalized_shape {
                anyhow::bail!(
                    "beta shape {:?} does not match normalized_shape {:?}",
                    beta_tensor.shape(),
                    normalized_shape
                );
            }
            let beta_broadcast = beta_tensor
                .data
                .broadcast(self.shape())
                .ok_or_else(|| anyhow::anyhow!("Failed to broadcast beta"))?;
            let shifted_data: Vec<T> = result
                .data
                .iter()
                .zip(beta_broadcast.iter())
                .map(|(&x, &b)| x + b)
                .collect();
            result = Self::from_vec(shifted_data, self.shape())?;
        }

        Ok(result)
    }

    /// Apply softmax function along a specific axis.
    ///
    /// Softmax normalizes values to a probability distribution along the specified axis.
    /// Uses numerically stable computation by subtracting the maximum value.
    ///
    /// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply softmax
    ///
    /// # Returns
    ///
    /// Tensor with same shape where values along axis sum to 1.0
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let logits = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let probs = logits.softmax(1).unwrap();
    /// assert_eq!(probs.shape(), &[2, 3]);
    ///
    /// // Each row should sum to 1.0
    /// let row0_sum = probs[&[0, 0]] + probs[&[0, 1]] + probs[&[0, 2]];
    /// assert!((row0_sum - 1.0).abs() < 1e-10);
    /// ```
    pub fn softmax(&self, axis: usize) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // For numerical stability, subtract max along axis
        let max_vals = self.max_axis(axis, true)?;
        let max_broadcast = max_vals
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast max values"))?;

        // Compute exp(x - max(x))
        let exp_data: Vec<T> = self
            .data
            .iter()
            .zip(max_broadcast.iter())
            .map(|(&x, &m)| (x - m).exp())
            .collect();
        let exp_tensor = Self::from_vec(exp_data, self.shape())?;

        // Sum along axis
        let sum_exp = exp_tensor.sum_axis(axis, true)?;
        let sum_broadcast = sum_exp
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast sum"))?;

        // Divide by sum
        let softmax_data: Vec<T> = exp_tensor
            .data
            .iter()
            .zip(sum_broadcast.iter())
            .map(|(&e, &s)| e / s)
            .collect();

        Self::from_vec(softmax_data, self.shape())
    }

    /// Apply log-softmax function along a specific axis.
    ///
    /// Log-softmax is numerically more stable than log(softmax(x)) for computing
    /// log probabilities.
    ///
    /// log_softmax(x_i) = (x_i - max(x)) - log(sum(exp(x_j - max(x))))
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply log-softmax
    ///
    /// # Returns
    ///
    /// Tensor with same shape containing log probabilities
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let logits = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0],
    ///     &[3]
    /// ).unwrap();
    ///
    /// let log_probs = logits.log_softmax(0).unwrap();
    /// assert_eq!(log_probs.shape(), &[3]);
    ///
    /// // exp(log_softmax) should equal softmax
    /// let probs = logits.softmax(0).unwrap();
    ///
    /// for i in 0..3 {
    ///     let prob_from_log = log_probs[&[i]].exp();
    ///     assert!((prob_from_log - probs[&[i]]).abs() < 1e-10);
    /// }
    /// ```
    pub fn log_softmax(&self, axis: usize) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // For numerical stability, subtract max along axis
        let max_vals = self.max_axis(axis, true)?;
        let max_broadcast = max_vals
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast max values"))?;

        // Compute x - max(x)
        let shifted_data: Vec<T> = self
            .data
            .iter()
            .zip(max_broadcast.iter())
            .map(|(&x, &m)| x - m)
            .collect();
        let shifted_tensor = Self::from_vec(shifted_data, self.shape())?;

        // Compute sum(exp(x - max(x)))
        let exp_data: Vec<T> = shifted_tensor.data.iter().map(|&x| x.exp()).collect();
        let exp_tensor = Self::from_vec(exp_data, self.shape())?;
        let sum_exp = exp_tensor.sum_axis(axis, true)?;

        // Compute log(sum(exp(x - max(x))))
        let log_sum_exp_data: Vec<T> = sum_exp.data.iter().map(|&x| x.ln()).collect();
        let log_sum_exp = Self::from_vec(log_sum_exp_data, sum_exp.shape())?;
        let log_sum_broadcast = log_sum_exp
            .data
            .broadcast(self.shape())
            .ok_or_else(|| anyhow::anyhow!("Failed to broadcast log_sum_exp"))?;

        // Final: (x - max(x)) - log(sum(exp(x - max(x))))
        let log_softmax_data: Vec<T> = shifted_tensor
            .data
            .iter()
            .zip(log_sum_broadcast.iter())
            .map(|(&x, &ls)| x - ls)
            .collect();

        Self::from_vec(log_softmax_data, self.shape())
    }

    /// Find the maximum value along a specific axis (helper for softmax).
    ///
    /// This is similar to the existing max method but works along an axis.
    fn max_axis(&self, axis: usize, keepdims: bool) -> anyhow::Result<Self>
    where
        T: PartialOrd,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // Calculate output shape
        let mut output_shape: Vec<usize> = self.shape().to_vec();
        output_shape.remove(axis);

        if output_shape.is_empty() {
            // Scalar result
            return Ok(Self::from_elem(&[1], self.max().clone()));
        }

        let output_size: usize = output_shape.iter().product();
        let axis_size = self.shape()[axis];

        let mut result_data = Vec::with_capacity(output_size);

        // Find max along axis
        for i in 0..output_size {
            let mut max_val = None;

            for j in 0..axis_size {
                // Convert flat index to multi-dimensional index
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = i;
                let mut axis_inserted = false;

                for (dim_idx, _) in self.shape().iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(j);
                        axis_inserted = true;
                    } else {
                        let actual_idx = if !axis_inserted && dim_idx < axis {
                            dim_idx
                        } else if axis_inserted && dim_idx > axis {
                            dim_idx - 1
                        } else {
                            dim_idx
                        };

                        let stride: usize = output_shape[actual_idx + 1..]
                            .iter()
                            .product::<usize>()
                            .max(1);
                        indices.push(remaining / stride);
                        remaining %= stride;
                    }
                }

                let val = &self.data[&indices[..]];
                max_val = Some(match max_val {
                    None => val.clone(),
                    Some(ref current_max) => {
                        if val > current_max {
                            val.clone()
                        } else {
                            current_max.clone()
                        }
                    }
                });
            }

            result_data.push(max_val.unwrap());
        }

        let mut result = Self::from_vec(result_data, &output_shape)?;
        if keepdims {
            result = result.unsqueeze(axis)?;
        }
        Ok(result)
    }
}
