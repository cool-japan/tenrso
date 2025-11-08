//! Dense tensor implementation with views and strides
//!
//! This module provides the core `DenseND<T>` type for dense N-dimensional
//! tensor storage, along with view types for zero-copy operations.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext` and `scirs2_core::random`.
//! Direct use of `ndarray`, `rand`, or `num_traits` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut, IxDyn};
use scirs2_core::numeric::{Num, NumCast};
use std::fmt;

/// Dense N-dimensional tensor backed by scirs2_core's ndarray
///
/// This is the primary dense tensor type in TenRSo, wrapping scirs2_core's
/// dynamic-dimensionality arrays with tensor-specific operations.
///
/// # Type Parameters
///
/// * `T` - The element type (typically `f32` or `f64`)
///
/// # Memory Layout
///
/// By default, tensors use C-contiguous (row-major) memory layout for
/// cache efficiency. This can be customized through scirs2_core's ndarray API.
///
/// # Examples
///
/// ```
/// use tenrso_core::dense::DenseND;
///
/// // Create a 3D tensor of zeros
/// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
/// assert_eq!(tensor.shape(), &[2, 3, 4]);
/// assert_eq!(tensor.rank(), 3);
/// ```
#[derive(Clone)]
pub struct DenseND<T> {
    /// Underlying ndarray storage (via scirs2_core)
    pub(crate) data: Array<T, IxDyn>,
}

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Create a tensor from an existing ndarray
    ///
    /// # Arguments
    ///
    /// * `array` - The source array with dynamic dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray_ext::Array;
    /// use tenrso_core::dense::DenseND;
    ///
    /// let arr = Array::<f64, _>::zeros(vec![2, 3]);
    /// let tensor = DenseND::from_array(arr);
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
    pub fn from_array(array: Array<T, IxDyn>) -> Self {
        Self { data: array }
    }

    /// Create a tensor from a vector with given shape
    ///
    /// # Arguments
    ///
    /// * `vec` - Flattened data in row-major order
    /// * `shape` - Target shape
    ///
    /// # Returns
    ///
    /// A tensor with the specified shape, or an error if dimensions don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = DenseND::from_vec(data, &[2, 3]).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 3]);
    /// ```
    pub fn from_vec(vec: Vec<T>, shape: &[usize]) -> anyhow::Result<Self> {
        let total: usize = shape.iter().product();
        if vec.len() != total {
            anyhow::bail!(
                "Shape {:?} requires {} elements, but got {}",
                shape,
                total,
                vec.len()
            );
        }
        let array = Array::from_shape_vec(IxDyn(shape), vec)?;
        Ok(Self { data: array })
    }

    /// Get the rank (number of dimensions) of this tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f32>::zeros(&[2, 3, 4]);
    /// assert_eq!(tensor.rank(), 3);
    /// ```
    pub fn rank(&self) -> usize {
        self.data.ndim()
    }

    /// Get the shape of this tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f32>::zeros(&[2, 3, 4]);
    /// assert_eq!(tensor.shape(), &[2, 3, 4]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the total number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f32>::zeros(&[2, 3, 4]);
    /// assert_eq!(tensor.len(), 24);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty (has zero elements)
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if the tensor is contiguous in memory.
    ///
    /// Contiguous tensors can be reshaped without copying.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// assert!(tensor.is_contiguous());
    ///
    /// // After permute, tensor may not be contiguous
    /// let permuted = tensor.permute(&[2, 0, 1]).unwrap();
    /// // Contiguity depends on the permutation
    /// ```
    pub fn is_contiguous(&self) -> bool {
        self.data.is_standard_layout()
    }

    /// Check if this is a square matrix (2D tensor with equal dimensions).
    ///
    /// Returns `true` only for 2D tensors where both dimensions are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let square = DenseND::<f64>::zeros(&[5, 5]);
    /// assert!(square.is_square());
    ///
    /// let rect = DenseND::<f64>::zeros(&[3, 5]);
    /// assert!(!rect.is_square());
    ///
    /// let tensor_3d = DenseND::<f64>::zeros(&[5, 5, 5]);
    /// assert!(!tensor_3d.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        self.rank() == 2 && self.shape()[0] == self.shape()[1]
    }

    /// Get a copy of the shape as a vector.
    ///
    /// This is useful when you need an owned copy of the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let shape_vec = tensor.shape_vec();
    /// assert_eq!(shape_vec, vec![2, 3, 4]);
    /// ```
    pub fn shape_vec(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    /// Get an immutable reference to the underlying ndarray
    pub fn as_array(&self) -> &Array<T, IxDyn> {
        &self.data
    }

    /// Get a mutable reference to the underlying ndarray
    pub fn as_array_mut(&mut self) -> &mut Array<T, IxDyn> {
        &mut self.data
    }

    /// Get an immutable view of the tensor
    pub fn view(&self) -> ArrayView<'_, T, IxDyn> {
        self.data.view()
    }

    /// Get a mutable view of the tensor
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T, IxDyn> {
        self.data.view_mut()
    }

    /// Create a tensor filled with a specific value
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
    /// let tensor = DenseND::from_elem(&[2, 3], 5.0);
    /// assert_eq!(tensor[&[0, 0]], 5.0);
    /// assert_eq!(tensor[&[1, 2]], 5.0);
    /// ```
    pub fn from_elem(shape: &[usize], value: T) -> Self {
        Self {
            data: Array::from_elem(IxDyn(shape), value),
        }
    }

    /// Create a tensor of zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// assert_eq!(tensor[&[0, 0, 0]], 0.0);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: Array::zeros(IxDyn(shape)),
        }
    }

    /// Create a tensor of ones
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[2, 3]);
    /// assert_eq!(tensor[&[0, 0]], 1.0);
    /// assert_eq!(tensor[&[1, 2]], 1.0);
    /// ```
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: Array::ones(IxDyn(shape)),
        }
    }
}

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
        // Use scirs2_core's quick API for Box-Muller normal distribution
        use scirs2_core::random::quick::random_f64;
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total / 2 * 2)
            .step_by(2)
            .flat_map(|_| {
                // Box-Muller transform for normal distribution
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
    /// Reshape the tensor to a new shape
    ///
    /// This operation is zero-copy when the tensor is contiguous.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The target shape
    ///
    /// # Returns
    ///
    /// A reshaped tensor, or an error if the total size doesn't match
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let reshaped = tensor.reshape(&[6, 4]).unwrap();
    /// assert_eq!(reshaped.shape(), &[6, 4]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> anyhow::Result<Self> {
        let new_size: usize = new_shape.iter().product();
        let old_size = self.len();

        if new_size != old_size {
            anyhow::bail!(
                "Cannot reshape tensor of size {} into shape {:?} (size {})",
                old_size,
                new_shape,
                new_size
            );
        }

        // Try zero-copy reshape first
        if let Ok(reshaped) = self.data.view().into_shape_with_order(IxDyn(new_shape)) {
            Ok(Self {
                data: reshaped.to_owned(),
            })
        } else {
            // Fall back to copy if not contiguous
            let flat: Vec<T> = self.data.iter().cloned().collect();
            Ok(Self {
                data: Array::from_shape_vec(IxDyn(new_shape), flat)?,
            })
        }
    }

    /// Permute the axes of the tensor
    ///
    /// # Arguments
    ///
    /// * `axes` - The new axis ordering
    ///
    /// # Returns
    ///
    /// A tensor with permuted axes
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let permuted = tensor.permute(&[2, 0, 1]).unwrap();
    /// assert_eq!(permuted.shape(), &[4, 2, 3]);
    /// ```
    pub fn permute(&self, axes: &[usize]) -> anyhow::Result<Self> {
        if axes.len() != self.rank() {
            anyhow::bail!(
                "Permutation axes length ({}) must match tensor rank ({})",
                axes.len(),
                self.rank()
            );
        }

        // Validate that axes is a valid permutation
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        for (i, &ax) in sorted.iter().enumerate() {
            if ax != i {
                anyhow::bail!("Invalid permutation: {:?}", axes);
            }
        }

        Ok(Self {
            data: self.data.clone().permuted_axes(IxDyn(axes)),
        })
    }

    /// Unfold the tensor along a specified mode (matricization)
    ///
    /// This operation rearranges the tensor into a matrix where the fibers
    /// along the specified mode become the rows.
    ///
    /// # Arguments
    ///
    /// * `mode` - The mode to unfold along (0-indexed)
    ///
    /// # Returns
    ///
    /// A 2D array with shape (shape\[mode\], product of other dimensions)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let unfolded = tensor.unfold(1).unwrap();
    /// assert_eq!(unfolded.shape(), &[3, 8]); // 8 = 2 * 4
    /// ```
    pub fn unfold(&self, mode: usize) -> anyhow::Result<scirs2_core::ndarray_ext::Array2<T>> {
        if mode >= self.rank() {
            anyhow::bail!(
                "Mode {} out of bounds for tensor with rank {}",
                mode,
                self.rank()
            );
        }

        let shape = self.shape();
        let mode_size = shape[mode];
        let other_size: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, &s)| s)
            .product();

        // Create permutation: [mode, 0, 1, ..., mode-1, mode+1, ..., rank-1]
        let mut perm: Vec<usize> = Vec::with_capacity(self.rank());
        perm.push(mode);
        for i in 0..self.rank() {
            if i != mode {
                perm.push(i);
            }
        }

        // Permute and reshape
        // Note: permuted_axes creates a non-contiguous view, so we need to make it contiguous
        let permuted = self.data.clone().permuted_axes(IxDyn(&perm));
        let contiguous = permuted.as_standard_layout().into_owned();
        let unfolded = contiguous.into_shape_with_order((mode_size, other_size))?;

        Ok(unfolded)
    }

    /// Fold a matrix back into a tensor along a specified mode
    ///
    /// This is the inverse operation of `unfold`.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The unfolded matrix
    /// * `shape` - The target tensor shape
    /// * `mode` - The mode that was used for unfolding
    ///
    /// # Returns
    ///
    /// A tensor with the specified shape
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray_ext::Array2;
    /// use tenrso_core::dense::DenseND;
    ///
    /// let matrix = Array2::<f64>::zeros((3, 8));
    /// let tensor = DenseND::fold(&matrix, &[2, 3, 4], 1).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 3, 4]);
    /// ```
    pub fn fold(
        matrix: &scirs2_core::ndarray_ext::Array2<T>,
        shape: &[usize],
        mode: usize,
    ) -> anyhow::Result<Self> {
        if mode >= shape.len() {
            anyhow::bail!("Mode {} out of bounds for shape {:?}", mode, shape);
        }

        let mode_size = shape[mode];
        let other_size: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, &s)| s)
            .product();

        if matrix.shape() != [mode_size, other_size] {
            anyhow::bail!(
                "Matrix shape {:?} incompatible with tensor shape {:?} and mode {}",
                matrix.shape(),
                shape,
                mode
            );
        }

        // Create intermediate shape: [mode_size, other_dims...]
        let mut inter_shape = Vec::with_capacity(shape.len());
        inter_shape.push(mode_size);
        for (i, &s) in shape.iter().enumerate() {
            if i != mode {
                inter_shape.push(s);
            }
        }

        // Reshape matrix to intermediate shape
        let inter = matrix.clone().into_shape_with_order(IxDyn(&inter_shape))?;

        // Create inverse permutation
        let mut inv_perm = vec![0; shape.len()];
        inv_perm[mode] = 0;
        let mut idx = 1;
        for (i, _) in shape.iter().enumerate() {
            if i != mode {
                inv_perm[i] = idx;
                idx += 1;
            }
        }

        // Permute back to original axis order
        let tensor = inter.permuted_axes(IxDyn(&inv_perm));

        Ok(Self { data: tensor })
    }

    /// Get an element at the specified index without panicking.
    ///
    /// Returns `None` if the index is out of bounds or has incorrect dimensionality.
    ///
    /// # Complexity
    ///
    /// O(1) - direct array access with bounds checking
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_elem(&[3, 4], 5.0);
    ///
    /// // Valid access
    /// assert_eq!(tensor.get(&[0, 0]), Some(&5.0));
    /// assert_eq!(tensor.get(&[2, 3]), Some(&5.0));
    ///
    /// // Out of bounds
    /// assert_eq!(tensor.get(&[10, 10]), None);
    ///
    /// // Wrong dimensionality
    /// assert_eq!(tensor.get(&[0]), None);
    /// assert_eq!(tensor.get(&[0, 0, 0]), None);
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        // Check dimensionality
        if index.len() != self.rank() {
            return None;
        }

        // Check bounds for each dimension
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape()[i] {
                return None;
            }
        }

        // Safe to index
        Some(&self.data[IxDyn(index)])
    }

    /// Get a mutable reference to an element at the specified index without panicking.
    ///
    /// Returns `None` if the index is out of bounds or has incorrect dimensionality.
    ///
    /// # Complexity
    ///
    /// O(1) - direct array access with bounds checking
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::zeros(&[3, 4]);
    ///
    /// // Valid access and modification
    /// if let Some(elem) = tensor.get_mut(&[1, 2]) {
    ///     *elem = 42.0;
    /// }
    /// assert_eq!(tensor[&[1, 2]], 42.0);
    ///
    /// // Out of bounds - no panic
    /// assert_eq!(tensor.get_mut(&[10, 10]), None);
    ///
    /// // Wrong dimensionality
    /// assert_eq!(tensor.get_mut(&[0]), None);
    /// ```
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        // Check dimensionality
        if index.len() != self.rank() {
            return None;
        }

        // Check bounds for each dimension
        let shape = self.shape().to_vec(); // Store shape to avoid borrow issues
        for (i, &idx) in index.iter().enumerate() {
            if idx >= shape[i] {
                return None;
            }
        }

        // Safe to index
        Some(&mut self.data[IxDyn(index)])
    }

    /// Get a reference to the underlying data as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[2, 3]);
    /// let slice = tensor.as_slice();
    /// assert_eq!(slice.len(), 6);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice().expect("Data should be contiguous")
    }

    /// Remove all singleton dimensions (dimensions with size 1).
    ///
    /// Returns a new tensor with all dimensions of size 1 removed.
    ///
    /// # Complexity
    ///
    /// O(1) - creates a view with modified shape, no data copying
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 1, 3, 1, 4]);
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed.shape(), &[2, 3, 4]);
    ///
    /// let tensor = DenseND::<f64>::zeros(&[1, 1, 5]);
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed.shape(), &[5]);
    ///
    /// // If no singleton dimensions, returns clone
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed.shape(), &[2, 3, 4]);
    /// ```
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&s| s != 1).copied().collect();

        // If all dimensions were 1, result should be a scalar (1D with size 1)
        let new_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };

        // Reshape will handle the conversion
        self.reshape(&new_shape)
            .expect("Squeeze should always produce valid shape")
    }

    /// Remove a specific singleton dimension.
    ///
    /// Returns an error if the specified axis doesn't have size 1.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove (must have size 1)
    ///
    /// # Complexity
    ///
    /// O(1) - creates a view with modified shape, no data copying
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The axis is out of bounds
    /// - The dimension at the specified axis is not 1
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 1, 3, 4]);
    /// let squeezed = tensor.squeeze_axis(1).unwrap();
    /// assert_eq!(squeezed.shape(), &[2, 3, 4]);
    ///
    /// // Error if dimension is not 1
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// assert!(tensor.squeeze_axis(1).is_err());
    /// ```
    pub fn squeeze_axis(&self, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if self.shape()[axis] != 1 {
            anyhow::bail!(
                "Cannot squeeze axis {} with size {} (must be 1)",
                axis,
                self.shape()[axis]
            );
        }

        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        // If removing the only dimension, result is a scalar (1D with size 1)
        let new_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };

        self.reshape(&new_shape)
    }

    /// Add a singleton dimension at the specified position.
    ///
    /// # Arguments
    ///
    /// * `axis` - The position to insert the new dimension (0 to rank, inclusive)
    ///
    /// # Complexity
    ///
    /// O(1) - creates a view with modified shape, no data copying
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of bounds (> rank).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    ///
    /// // Add dimension at beginning
    /// let unsqueezed = tensor.unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[1, 2, 3, 4]);
    ///
    /// // Add dimension in middle
    /// let unsqueezed = tensor.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[2, 1, 3, 4]);
    ///
    /// // Add dimension at end
    /// let unsqueezed = tensor.unsqueeze(3).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[2, 3, 4, 1]);
    /// ```
    pub fn unsqueeze(&self, axis: usize) -> anyhow::Result<Self> {
        if axis > self.rank() {
            anyhow::bail!(
                "Axis {} out of bounds for unsqueeze (max {})",
                axis,
                self.rank()
            );
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);

        self.reshape(&new_shape)
    }

    /// Concatenate tensors along an existing axis.
    ///
    /// All tensors must have the same shape except along the concatenation axis.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to concatenate (must include at least one tensor)
    /// * `axis` - The axis along which to concatenate
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements in all tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor list is empty
    /// - The axis is out of bounds
    /// - Tensors have incompatible shapes (differ on non-concatenation axes)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::ones(&[2, 3]);
    /// let b = DenseND::<f64>::zeros(&[2, 3]);
    /// let c = DenseND::<f64>::from_elem(&[2, 3], 5.0);
    ///
    /// // Concatenate along axis 0 (rows)
    /// let concat = DenseND::concatenate(&[a.clone(), b.clone()], 0).unwrap();
    /// assert_eq!(concat.shape(), &[4, 3]);
    ///
    /// // Concatenate along axis 1 (columns)
    /// let concat = DenseND::concatenate(&[a, b, c], 1).unwrap();
    /// assert_eq!(concat.shape(), &[2, 9]);
    /// ```
    pub fn concatenate(tensors: &[Self], axis: usize) -> anyhow::Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot concatenate empty list of tensors");
        }

        let first = &tensors[0];
        let rank = first.rank();

        if axis >= rank {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, rank);
        }

        // Verify all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.rank() != rank {
                anyhow::bail!(
                    "Tensor {} has rank {} but expected {}",
                    i,
                    tensor.rank(),
                    rank
                );
            }

            for (ax, (&s1, &s2)) in first.shape().iter().zip(tensor.shape().iter()).enumerate() {
                if ax != axis && s1 != s2 {
                    anyhow::bail!(
                        "Tensor {} has incompatible shape at axis {}: {} vs {}",
                        i,
                        ax,
                        s2,
                        s1
                    );
                }
            }
        }

        // Compute output shape
        let concat_size: usize = tensors.iter().map(|t| t.shape()[axis]).sum();
        let mut output_shape = first.shape().to_vec();
        output_shape[axis] = concat_size;

        // Create output tensor
        let mut output = Self::zeros(&output_shape);

        // Copy data from each tensor
        let mut offset = 0;
        for tensor in tensors {
            let size = tensor.shape()[axis];

            // Use ndarray slicing to copy data
            for idx in 0..size {
                // This is a simplified approach - in production, we'd use more efficient slicing
                // For now, copy element by element (can be optimized later)
                let src_indices: Vec<Vec<usize>> = generate_indices(tensor.shape(), axis, idx);
                let dst_indices: Vec<Vec<usize>> =
                    generate_indices(&output_shape, axis, offset + idx);

                for (src_idx, dst_idx) in src_indices.iter().zip(dst_indices.iter()) {
                    output[&dst_idx[..]] = tensor[&src_idx[..]].clone();
                }
            }

            offset += size;
        }

        Ok(output)
    }

    /// Stack tensors along a new axis.
    ///
    /// All tensors must have exactly the same shape. A new dimension is added
    /// at the specified position.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to stack (must include at least one tensor)
    /// * `axis` - The position of the new axis (0 to rank, inclusive)
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements in all tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor list is empty
    /// - The axis is out of bounds (> rank)
    /// - Tensors have different shapes
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::ones(&[2, 3]);
    /// let b = DenseND::<f64>::zeros(&[2, 3]);
    /// let c = DenseND::<f64>::from_elem(&[2, 3], 5.0);
    ///
    /// // Stack along new first axis
    /// let stacked = DenseND::stack(&[a.clone(), b.clone(), c.clone()], 0).unwrap();
    /// assert_eq!(stacked.shape(), &[3, 2, 3]);
    ///
    /// // Stack along new last axis
    /// let stacked = DenseND::stack(&[a, b, c], 2).unwrap();
    /// assert_eq!(stacked.shape(), &[2, 3, 3]);
    /// ```
    pub fn stack(tensors: &[Self], axis: usize) -> anyhow::Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot stack empty list of tensors");
        }

        let first = &tensors[0];
        let rank = first.rank();

        if axis > rank {
            anyhow::bail!("Axis {} out of bounds for stack (max {})", axis, rank);
        }

        // Verify all tensors have the same shape
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first.shape() {
                anyhow::bail!(
                    "Tensor {} has shape {:?} but expected {:?}",
                    i,
                    tensor.shape(),
                    first.shape()
                );
            }
        }

        // Add singleton dimension to each tensor, then concatenate
        let unsqueezed: Result<Vec<Self>, _> = tensors.iter().map(|t| t.unsqueeze(axis)).collect();
        let unsqueezed = unsqueezed?;

        Self::concatenate(&unsqueezed, axis)
    }

    /// Split a tensor into multiple tensors along an axis.
    ///
    /// # Arguments
    ///
    /// * `num_splits` - Number of equal-sized splits to create
    /// * `axis` - The axis along which to split
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The axis is out of bounds
    /// - The axis size is not evenly divisible by num_splits
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[6, 4]);
    ///
    /// // Split into 3 equal parts along axis 0
    /// let splits = tensor.split(3, 0).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// assert_eq!(splits[0].shape(), &[2, 4]);
    ///
    /// // Split into 2 parts along axis 1
    /// let splits = tensor.split(2, 1).unwrap();
    /// assert_eq!(splits.len(), 2);
    /// assert_eq!(splits[0].shape(), &[6, 2]);
    /// ```
    pub fn split(&self, num_splits: usize, axis: usize) -> anyhow::Result<Vec<Self>> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let axis_size = self.shape()[axis];
        if !axis_size.is_multiple_of(num_splits) {
            anyhow::bail!(
                "Axis size {} is not evenly divisible by num_splits {}",
                axis_size,
                num_splits
            );
        }

        let chunk_size = axis_size / num_splits;
        let mut result = Vec::with_capacity(num_splits);

        for i in 0..num_splits {
            let start = i * chunk_size;
            let end = start + chunk_size;

            // Extract slice along axis
            let mut new_shape = self.shape().to_vec();
            new_shape[axis] = chunk_size;
            let mut chunk = Self::zeros(&new_shape);

            // Copy data for this chunk
            for idx in start..end {
                let src_indices = generate_indices(self.shape(), axis, idx);
                let dst_indices = generate_indices(&new_shape, axis, idx - start);

                for (src_idx, dst_idx) in src_indices.iter().zip(dst_indices.iter()) {
                    chunk[&dst_idx[..]] = self[&src_idx[..]].clone();
                }
            }

            result.push(chunk);
        }

        Ok(result)
    }

    /// Split a tensor into chunks of specified size along an axis.
    ///
    /// The last chunk may be smaller if the axis size is not evenly divisible.
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Size of each chunk
    /// * `axis` - The axis along which to chunk
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The axis is out of bounds
    /// - chunk_size is 0
    /// - chunk_size is larger than the axis size
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[7, 4]);
    ///
    /// // Chunk into pieces of size 3 along axis 0
    /// let chunks = tensor.chunk(3, 0).unwrap();
    /// assert_eq!(chunks.len(), 3);
    /// assert_eq!(chunks[0].shape(), &[3, 4]);
    /// assert_eq!(chunks[1].shape(), &[3, 4]);
    /// assert_eq!(chunks[2].shape(), &[1, 4]); // Last chunk is smaller
    /// ```
    pub fn chunk(&self, chunk_size: usize, axis: usize) -> anyhow::Result<Vec<Self>> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if chunk_size == 0 {
            anyhow::bail!("Chunk size must be greater than 0");
        }

        let axis_size = self.shape()[axis];
        if chunk_size > axis_size {
            anyhow::bail!(
                "Chunk size {} is larger than axis size {}",
                chunk_size,
                axis_size
            );
        }

        let num_chunks = axis_size.div_ceil(chunk_size);
        let mut result = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, axis_size);
            let current_chunk_size = end - start;

            // Extract slice along axis
            let mut new_shape = self.shape().to_vec();
            new_shape[axis] = current_chunk_size;
            let mut chunk = Self::zeros(&new_shape);

            // Copy data for this chunk
            for idx in start..end {
                let src_indices = generate_indices(self.shape(), axis, idx);
                let dst_indices = generate_indices(&new_shape, axis, idx - start);

                for (src_idx, dst_idx) in src_indices.iter().zip(dst_indices.iter()) {
                    chunk[&dst_idx[..]] = self[&src_idx[..]].clone();
                }
            }

            result.push(chunk);
        }

        Ok(result)
    }

    /// Transpose a 2D matrix.
    ///
    /// This is a specialized operation for 2D tensors that swaps rows and columns.
    /// For general axis permutation, use `permute`.
    ///
    /// # Complexity
    ///
    /// O(1) - creates a view with swapped axes, no data copying
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let transposed = tensor.transpose().unwrap();
    ///
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// assert_eq!(transposed[&[0, 0]], 1.0);
    /// assert_eq!(transposed[&[0, 1]], 4.0);
    /// ```
    pub fn transpose(&self) -> anyhow::Result<Self> {
        if self.rank() != 2 {
            anyhow::bail!(
                "Transpose is only defined for 2D tensors, got rank {}",
                self.rank()
            );
        }

        self.permute(&[1, 0])
    }

    /// Matrix multiplication for 2D tensors.
    ///
    /// Computes the matrix product C = AB where A and B are 2D tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand matrix
    ///
    /// # Complexity
    ///
    /// O(n³) for n×n matrices (uses standard matrix multiplication)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either tensor is not 2D
    /// - The inner dimensions don't match (A's columns != B's rows)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    ///
    /// let c = a.matmul(&b).unwrap();
    /// assert_eq!(c.shape(), &[2, 2]);
    /// // [[1*5 + 2*7, 1*6 + 2*8],
    /// //  [3*5 + 4*7, 3*6 + 4*8]]
    /// // = [[19, 22], [43, 50]]
    /// assert_eq!(c[&[0, 0]], 19.0);
    /// assert_eq!(c[&[0, 1]], 22.0);
    /// assert_eq!(c[&[1, 0]], 43.0);
    /// assert_eq!(c[&[1, 1]], 50.0);
    /// ```
    pub fn matmul(&self, other: &Self) -> anyhow::Result<Self>
    where
        T: std::ops::Mul<Output = T> + std::iter::Sum,
    {
        if self.rank() != 2 || other.rank() != 2 {
            anyhow::bail!("Matrix multiplication requires 2D tensors");
        }

        let (m, k1) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);

        if k1 != k2 {
            anyhow::bail!(
                "Matrix dimensions incompatible: ({}, {}) × ({}, {})",
                m,
                k1,
                k2,
                n
            );
        }

        // Manual matrix multiplication
        let mut result = Self::zeros(&[m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..k1 {
                    sum = sum + self[&[i, k]].clone() * other[&[k, j]].clone();
                }
                result[&[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// Element-wise multiplication (Hadamard product) with another tensor.
    ///
    /// Computes C[i,j,...] = A[i,j,...] * B[i,j,...] for all indices.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply element-wise
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::from_elem(&[2, 3], 2.0);
    /// let b = DenseND::<f64>::from_elem(&[2, 3], 3.0);
    ///
    /// let c = a.hadamard(&b).unwrap();
    /// assert_eq!(c[&[0, 0]], 6.0);
    /// assert_eq!(c[&[1, 2]], 6.0);
    /// ```
    pub fn hadamard(&self, other: &Self) -> anyhow::Result<Self>
    where
        T: std::ops::Mul<Output = T>,
    {
        if self.shape() != other.shape() {
            anyhow::bail!(
                "Shape mismatch for Hadamard product: {:?} vs {:?}",
                self.shape(),
                other.shape()
            );
        }

        let result_data = &self.data * &other.data;
        Ok(Self { data: result_data })
    }

    /// Extract the diagonal of a 2D matrix.
    ///
    /// Returns a 1D tensor containing the diagonal elements.
    ///
    /// # Complexity
    ///
    /// O(min(m, n)) where m×n is the matrix shape
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     &[3, 3]
    /// ).unwrap();
    ///
    /// let diag = tensor.diagonal().unwrap();
    /// assert_eq!(diag.shape(), &[3]);
    /// assert_eq!(diag[&[0]], 1.0);
    /// assert_eq!(diag[&[1]], 5.0);
    /// assert_eq!(diag[&[2]], 9.0);
    /// ```
    pub fn diagonal(&self) -> anyhow::Result<Self> {
        if self.rank() != 2 {
            anyhow::bail!(
                "Diagonal is only defined for 2D tensors, got rank {}",
                self.rank()
            );
        }

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let diag_len = std::cmp::min(rows, cols);

        let mut diag_vec = Vec::with_capacity(diag_len);
        for i in 0..diag_len {
            diag_vec.push(self[&[i, i]].clone());
        }

        Self::from_vec(diag_vec, &[diag_len])
    }

    /// Compute the trace of a square matrix.
    ///
    /// The trace is the sum of diagonal elements.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the matrix dimension
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not a square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     &[3, 3]
    /// ).unwrap();
    ///
    /// let trace = tensor.trace().unwrap();
    /// assert_eq!(trace, 15.0); // 1 + 5 + 9
    /// ```
    pub fn trace(&self) -> anyhow::Result<T>
    where
        T: std::iter::Sum,
    {
        if !self.is_square() {
            anyhow::bail!(
                "Trace is only defined for square matrices, got shape {:?}",
                self.shape()
            );
        }

        let n = self.shape()[0];
        Ok((0..n).map(|i| self[&[i, i]].clone()).sum())
    }

    /// Broadcast this tensor to a target shape
    ///
    /// Broadcasting allows operations between tensors with compatible but different shapes.
    /// Dimensions are compatible if they are equal or one of them is 1.
    ///
    /// # Broadcasting Rules
    ///
    /// 1. Compare shapes from right to left
    /// 2. Dimensions are compatible if they are equal or one of them is 1
    /// 3. Missing dimensions are treated as 1
    /// 4. The result shape is the maximum along each dimension
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The shape to broadcast to
    ///
    /// # Returns
    ///
    /// A new tensor with the target shape, or an error if shapes are incompatible
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of elements in the target shape
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // Broadcast (3, 1) to (3, 4)
    /// let tensor = DenseND::<f64>::ones(&[3, 1]);
    /// let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
    /// assert_eq!(broadcast.shape(), &[3, 4]);
    ///
    /// // Broadcast (1,) to (3, 4)
    /// let tensor = DenseND::<f64>::from_vec(vec![2.0], &[1]).unwrap();
    /// let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
    /// assert_eq!(broadcast.shape(), &[3, 4]);
    /// for i in 0..3 {
    ///     for j in 0..4 {
    ///         assert_eq!(broadcast[&[i, j]], 2.0);
    ///     }
    /// }
    /// ```
    pub fn broadcast_to(&self, target_shape: &[usize]) -> anyhow::Result<Self> {
        let self_shape = self.shape();

        // Check if broadcasting is possible
        if !shapes_broadcastable(self_shape, target_shape) {
            anyhow::bail!(
                "Shapes {:?} and {:?} are not broadcastable",
                self_shape,
                target_shape
            );
        }

        // If shapes already match, return a clone
        if self_shape == target_shape {
            return Ok(self.clone());
        }

        // Create result tensor
        let mut result = Self::zeros(target_shape);

        // Copy data with broadcasting
        broadcast_copy(&self.data, &mut result.data, self_shape, target_shape)?;

        Ok(result)
    }
}

// Helper function to check if two shapes are broadcastable
fn shapes_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);

    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }
    true
}

// Helper function to compute broadcast shape
fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    if !shapes_broadcastable(shape1, shape2) {
        return None;
    }

    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

        result.push(dim1.max(dim2));
    }

    result.reverse();
    Some(result)
}

// Helper function to copy data with broadcasting
fn broadcast_copy<T>(
    src: &Array<T, IxDyn>,
    dst: &mut Array<T, IxDyn>,
    src_shape: &[usize],
    dst_shape: &[usize],
) -> anyhow::Result<()>
where
    T: Clone + Num,
{
    let src_rank = src_shape.len();
    let dst_rank = dst_shape.len();
    let rank_diff = dst_rank.saturating_sub(src_rank);

    // Helper to map destination index to source index
    let map_index = |dst_idx: &[usize]| -> Vec<usize> {
        let mut src_idx = Vec::with_capacity(src_rank);
        for (i, &src_dim) in src_shape.iter().enumerate() {
            let dst_dim_idx = rank_diff + i;
            let dst_val = dst_idx[dst_dim_idx];
            // If source dimension is 1, use index 0 (broadcasting)
            src_idx.push(if src_dim == 1 { 0 } else { dst_val });
        }
        src_idx
    };

    // Iterate through all destination indices
    let total_elements: usize = dst_shape.iter().product();
    for flat_idx in 0..total_elements {
        // Convert flat index to multi-dimensional index
        let mut dst_idx = Vec::with_capacity(dst_rank);
        let mut remaining = flat_idx;
        for i in (0..dst_rank).rev() {
            let dim_size = dst_shape[i];
            dst_idx.insert(0, remaining % dim_size);
            remaining /= dim_size;
        }

        // Map to source index and copy
        let src_idx = map_index(&dst_idx);
        dst[IxDyn(&dst_idx)] = src[IxDyn(&src_idx)].clone();
    }

    Ok(())
}

// Helper function to generate all indices for copying slices during concatenation
fn generate_indices(shape: &[usize], axis: usize, axis_value: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut indices = vec![0; shape.len()];
    indices[axis] = axis_value;

    fn recurse(
        shape: &[usize],
        axis: usize,
        indices: &mut [usize],
        current_dim: usize,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current_dim == shape.len() {
            result.push(indices.to_vec());
            return;
        }

        if current_dim == axis {
            recurse(shape, axis, indices, current_dim + 1, result);
        } else {
            for i in 0..shape[current_dim] {
                indices[current_dim] = i;
                recurse(shape, axis, indices, current_dim + 1, result);
            }
        }
    }

    recurse(shape, axis, &mut indices, 0, &mut result);
    result
}

impl<T> DenseND<T>
where
    T: Clone + Num + NumCast + std::ops::Mul<Output = T> + std::iter::Sum,
{
    /// Compute the Frobenius norm of the tensor
    ///
    /// The Frobenius norm is the square root of the sum of squares of all elements.
    /// For a tensor X, this is: ||X||_F = sqrt(Σᵢⱼₖ... X²ᵢⱼₖ...)
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
}

impl<T> std::ops::Index<&[usize]> for DenseND<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.data[IxDyn(index)]
    }
}

impl<T> std::ops::IndexMut<&[usize]> for DenseND<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(index)]
    }
}

impl<'b, T> std::ops::Sub<&'b DenseND<T>> for &DenseND<T>
where
    T: Clone + Num,
{
    type Output = DenseND<T>;

    fn sub(self, rhs: &'b DenseND<T>) -> Self::Output {
        // Try to broadcast shapes
        let target_shape = broadcast_shape(self.shape(), rhs.shape())
            .expect("Shapes are not broadcastable for subtraction");

        // Broadcast both operands to target shape if needed
        let lhs_broadcast = if self.shape() != target_shape.as_slice() {
            self.broadcast_to(&target_shape)
                .expect("Failed to broadcast left operand")
        } else {
            self.clone()
        };

        let rhs_broadcast = if rhs.shape() != target_shape.as_slice() {
            rhs.broadcast_to(&target_shape)
                .expect("Failed to broadcast right operand")
        } else {
            rhs.clone()
        };

        let result = &lhs_broadcast.data - &rhs_broadcast.data;
        DenseND { data: result }
    }
}

impl<'b, T> std::ops::Add<&'b DenseND<T>> for &DenseND<T>
where
    T: Clone + Num,
{
    type Output = DenseND<T>;

    fn add(self, rhs: &'b DenseND<T>) -> Self::Output {
        // Try to broadcast shapes
        let target_shape = broadcast_shape(self.shape(), rhs.shape())
            .expect("Shapes are not broadcastable for addition");

        // Broadcast both operands to target shape if needed
        let lhs_broadcast = if self.shape() != target_shape.as_slice() {
            self.broadcast_to(&target_shape)
                .expect("Failed to broadcast left operand")
        } else {
            self.clone()
        };

        let rhs_broadcast = if rhs.shape() != target_shape.as_slice() {
            rhs.broadcast_to(&target_shape)
                .expect("Failed to broadcast right operand")
        } else {
            rhs.clone()
        };

        let result = &lhs_broadcast.data + &rhs_broadcast.data;
        DenseND { data: result }
    }
}

// Scalar operations
impl<T> std::ops::Mul<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;

    fn mul(self, scalar: T) -> Self::Output {
        let result = &self.data * scalar;
        DenseND { data: result }
    }
}

impl<T> std::ops::Div<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;

    fn div(self, scalar: T) -> Self::Output {
        let result = &self.data / scalar;
        DenseND { data: result }
    }
}

impl<T> std::ops::Add<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;

    fn add(self, scalar: T) -> Self::Output {
        let result = &self.data + scalar;
        DenseND { data: result }
    }
}

impl<T> std::ops::Sub<T> for &DenseND<T>
where
    T: Clone + Num + scirs2_core::ndarray_ext::ScalarOperand,
{
    type Output = DenseND<T>;

    fn sub(self, scalar: T) -> Self::Output {
        let result = &self.data - scalar;
        DenseND { data: result }
    }
}

impl<T: fmt::Debug + Clone + Num> fmt::Debug for DenseND<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseND")
            .field("shape", &self.shape())
            .field("rank", &self.rank())
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_zeros() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.len(), 24);
        assert_eq!(tensor[&[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_create_ones() {
        let tensor = DenseND::<f64>::ones(&[2, 3]);
        assert_eq!(tensor[&[0, 0]], 1.0);
        assert_eq!(tensor[&[1, 2]], 1.0);
    }

    #[test]
    fn test_from_elem() {
        let tensor = DenseND::from_elem(&[2, 3], 5.0);
        assert_eq!(tensor[&[0, 0]], 5.0);
        assert_eq!(tensor[&[1, 2]], 5.0);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = DenseND::from_vec(data, &[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor[&[0, 0]], 1.0);
        assert_eq!(tensor[&[1, 2]], 6.0);
    }

    #[test]
    fn test_reshape() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let reshaped = tensor.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.len(), 24);
    }

    #[test]
    fn test_permute() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let permuted = tensor.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), &[4, 2, 3]);
    }

    #[test]
    fn test_unfold() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unfolded = tensor.unfold(1).unwrap();
        assert_eq!(unfolded.shape(), &[3, 8]);
    }

    #[test]
    fn test_unfold_fold_roundtrip() {
        let shape = vec![2, 3, 4];
        let tensor = DenseND::<f64>::ones(&shape);

        for mode in 0..3 {
            let unfolded = tensor.unfold(mode).unwrap();
            let folded = DenseND::fold(&unfolded, &shape, mode).unwrap();
            assert_eq!(folded.shape(), tensor.shape());
            assert_eq!(folded.len(), tensor.len());
        }
    }

    #[test]
    fn test_random_uniform() {
        let tensor = DenseND::<f64>::random_uniform(&[10, 10], 0.0, 1.0);
        assert_eq!(tensor.shape(), &[10, 10]);

        // Check that values are in range
        for val in tensor.data.iter() {
            assert!(*val >= 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_random_normal() {
        let tensor = DenseND::<f64>::random_normal(&[10, 10], 0.0, 1.0);
        assert_eq!(tensor.shape(), &[10, 10]);
    }

    #[test]
    fn test_squeeze_all() {
        let tensor = DenseND::<f64>::zeros(&[2, 1, 3, 1, 4]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_squeeze_all_ones() {
        let tensor = DenseND::<f64>::zeros(&[1, 1, 1]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[1]);
    }

    #[test]
    fn test_squeeze_no_singletons() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_squeeze_axis() {
        let tensor = DenseND::<f64>::zeros(&[2, 1, 3, 4]);
        let squeezed = tensor.squeeze_axis(1).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_squeeze_axis_error() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        assert!(tensor.squeeze_axis(1).is_err());
    }

    #[test]
    fn test_unsqueeze_beginning() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_unsqueeze_middle() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 3, 4]);
    }

    #[test]
    fn test_unsqueeze_end() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(3).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 3, 4, 1]);
    }

    #[test]
    fn test_unsqueeze_squeeze_roundtrip() {
        let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
        let unsqueezed = tensor.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 3, 4]);

        let squeezed = unsqueezed.squeeze_axis(1).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_concatenate_axis_0() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);

        let concat = DenseND::concatenate(&[a, b], 0).unwrap();
        assert_eq!(concat.shape(), &[4, 3]);
    }

    #[test]
    fn test_concatenate_axis_1() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 4]);

        let concat = DenseND::concatenate(&[a, b], 1).unwrap();
        assert_eq!(concat.shape(), &[2, 7]);
    }

    #[test]
    fn test_concatenate_multiple() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let c = DenseND::<f64>::from_elem(&[2, 3], 5.0);

        let concat = DenseND::concatenate(&[a, b, c], 0).unwrap();
        assert_eq!(concat.shape(), &[6, 3]);
    }

    #[test]
    fn test_stack_axis_0() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);

        let stacked = DenseND::stack(&[a, b], 0).unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_stack_axis_1() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);

        let stacked = DenseND::stack(&[a, b], 1).unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_stack_axis_2() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[2, 3]);
        let c = DenseND::<f64>::from_elem(&[2, 3], 5.0);

        let stacked = DenseND::stack(&[a, b, c], 2).unwrap();
        assert_eq!(stacked.shape(), &[2, 3, 3]);
    }

    #[test]
    fn test_scalar_multiply() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 2.0);
        let result = &tensor * 3.0;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 6.0);
        assert_eq!(result[&[1, 2]], 6.0);
    }

    #[test]
    fn test_scalar_divide() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 12.0);
        let result = &tensor / 4.0;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 3.0);
        assert_eq!(result[&[1, 2]], 3.0);
    }

    #[test]
    fn test_scalar_add() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 2.0);
        let result = &tensor + 5.0;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 7.0);
        assert_eq!(result[&[1, 2]], 7.0);
    }

    #[test]
    fn test_scalar_subtract() {
        let tensor = DenseND::<f64>::from_elem(&[2, 3], 10.0);
        let result = &tensor - 3.0;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[&[0, 0]], 7.0);
        assert_eq!(result[&[1, 2]], 7.0);
    }

    #[test]
    fn test_split_axis_0() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let splits = tensor.split(3, 0).unwrap();

        assert_eq!(splits.len(), 3);
        for split in &splits {
            assert_eq!(split.shape(), &[2, 4]);
        }
    }

    #[test]
    fn test_split_axis_1() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let splits = tensor.split(2, 1).unwrap();

        assert_eq!(splits.len(), 2);
        for split in &splits {
            assert_eq!(split.shape(), &[6, 2]);
        }
    }

    #[test]
    fn test_split_error_not_divisible() {
        let tensor = DenseND::<f64>::ones(&[7, 4]);
        assert!(tensor.split(3, 0).is_err());
    }

    #[test]
    fn test_chunk_equal_size() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let chunks = tensor.chunk(2, 0).unwrap();

        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.shape(), &[2, 4]);
        }
    }

    #[test]
    fn test_chunk_last_smaller() {
        let tensor = DenseND::<f64>::ones(&[7, 4]);
        let chunks = tensor.chunk(3, 0).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[3, 4]);
        assert_eq!(chunks[1].shape(), &[3, 4]);
        assert_eq!(chunks[2].shape(), &[1, 4]);
    }

    #[test]
    fn test_split_concatenate_roundtrip() {
        let tensor = DenseND::<f64>::ones(&[6, 4]);
        let splits = tensor.split(3, 0).unwrap();
        let reconstructed = DenseND::concatenate(&splits, 0).unwrap();

        assert_eq!(reconstructed.shape(), tensor.shape());
    }

    #[test]
    fn test_transpose() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed[&[0, 0]], 1.0);
        assert_eq!(transposed[&[0, 1]], 4.0);
        assert_eq!(transposed[&[1, 0]], 2.0);
        assert_eq!(transposed[&[1, 1]], 5.0);
    }

    #[test]
    fn test_transpose_error_3d() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        assert!(tensor.transpose().is_err());
    }

    #[test]
    fn test_matmul() {
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[&[0, 0]], 19.0);
        assert_eq!(c[&[0, 1]], 22.0);
        assert_eq!(c[&[1, 0]], 43.0);
        assert_eq!(c[&[1, 1]], 50.0);
    }

    #[test]
    fn test_matmul_rectangular() {
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_matmul_error_mismatch() {
        let a = DenseND::<f64>::zeros(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[4, 2]);

        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_hadamard() {
        let a = DenseND::<f64>::from_elem(&[2, 3], 2.0);
        let b = DenseND::<f64>::from_elem(&[2, 3], 3.0);

        let c = a.hadamard(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c[&[0, 0]], 6.0);
        assert_eq!(c[&[1, 2]], 6.0);
    }

    #[test]
    fn test_hadamard_error_shape_mismatch() {
        let a = DenseND::<f64>::ones(&[2, 3]);
        let b = DenseND::<f64>::ones(&[3, 2]);

        assert!(a.hadamard(&b).is_err());
    }

    #[test]
    fn test_diagonal_square() {
        let tensor =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .unwrap();

        let diag = tensor.diagonal().unwrap();
        assert_eq!(diag.shape(), &[3]);
        assert_eq!(diag[&[0]], 1.0);
        assert_eq!(diag[&[1]], 5.0);
        assert_eq!(diag[&[2]], 9.0);
    }

    #[test]
    fn test_diagonal_rectangular() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let diag = tensor.diagonal().unwrap();
        assert_eq!(diag.shape(), &[2]);
        assert_eq!(diag[&[0]], 1.0);
        assert_eq!(diag[&[1]], 5.0);
    }

    #[test]
    fn test_trace() {
        let tensor =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
                .unwrap();

        let trace = tensor.trace().unwrap();
        assert_eq!(trace, 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_trace_error_non_square() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        assert!(tensor.trace().is_err());
    }

    // Broadcasting tests
    #[test]
    fn test_broadcast_to_simple() {
        let tensor = DenseND::<f64>::ones(&[3, 1]);
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);

        // Check all values are 1.0
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(broadcast[&[i, j]], 1.0);
            }
        }
    }

    #[test]
    fn test_broadcast_to_scalar() {
        let tensor = DenseND::<f64>::from_vec(vec![2.0], &[1]).unwrap();
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);

        // Check all values are 2.0
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(broadcast[&[i, j]], 2.0);
            }
        }
    }

    #[test]
    fn test_broadcast_to_3d() {
        let tensor = DenseND::<f64>::ones(&[1, 3, 1]);
        let broadcast = tensor.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[2, 3, 4]);

        // Check all values are 1.0
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(broadcast[&[i, j, k]], 1.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_to_rank_mismatch() {
        // Broadcasting (2,) to (3, 4) should work (prepend 1)
        let tensor = DenseND::<f64>::ones(&[4]);
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_to_incompatible() {
        let tensor = DenseND::<f64>::ones(&[3, 2]);
        let result = tensor.broadcast_to(&[3, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_to_same_shape() {
        let tensor = DenseND::<f64>::ones(&[3, 4]);
        let broadcast = tensor.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(broadcast.shape(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_add() {
        // (3, 1) + (1, 4) → (3, 4)
        let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[1, 4]).unwrap();

        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);

        // Expected results:
        // [1.0, 1.0, 1.0, 1.0]     [10, 20, 30, 40]     [11, 21, 31, 41]
        // [2.0, 2.0, 2.0, 2.0]  +  [10, 20, 30, 40]  =  [12, 22, 32, 42]
        // [3.0, 3.0, 3.0, 3.0]     [10, 20, 30, 40]     [13, 23, 33, 43]
        assert_eq!(result[&[0, 0]], 11.0);
        assert_eq!(result[&[0, 1]], 21.0);
        assert_eq!(result[&[1, 0]], 12.0);
        assert_eq!(result[&[2, 3]], 43.0);
    }

    #[test]
    fn test_broadcast_sub() {
        // (2, 3) - (3,) → (2, 3)
        let a =
            DenseND::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]).unwrap();
        let b = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = &a - &b;
        assert_eq!(result.shape(), &[2, 3]);

        // Expected: [[9, 18, 27], [39, 48, 57]]
        assert_eq!(result[&[0, 0]], 9.0);
        assert_eq!(result[&[0, 1]], 18.0);
        assert_eq!(result[&[0, 2]], 27.0);
        assert_eq!(result[&[1, 0]], 39.0);
    }

    #[test]
    fn test_broadcast_complex_3d() {
        // (2, 1, 4) + (3, 1) → (2, 3, 4)
        let a = DenseND::<f64>::ones(&[2, 1, 4]);
        let b = DenseND::<f64>::ones(&[3, 1]);

        let result = &a + &b;
        assert_eq!(result.shape(), &[2, 3, 4]);

        // All values should be 2.0 (1.0 + 1.0)
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(result[&[i, j, k]], 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_scalar_like() {
        // (1,) + (3, 4) → (3, 4)
        let a = DenseND::<f64>::from_vec(vec![5.0], &[1]).unwrap();
        let b = DenseND::<f64>::ones(&[3, 4]);

        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);

        // All values should be 6.0 (5.0 + 1.0)
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(result[&[i, j]], 6.0);
            }
        }
    }

    #[test]
    fn test_broadcast_matching_shapes() {
        // (3, 4) + (3, 4) → (3, 4) (no actual broadcasting)
        let a = DenseND::<f64>::ones(&[3, 4]);
        let b = DenseND::<f64>::ones(&[3, 4]);

        let result = &a + &b;
        assert_eq!(result.shape(), &[3, 4]);

        // All values should be 2.0
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(result[&[i, j]], 2.0);
            }
        }
    }
}
