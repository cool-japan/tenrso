//! Dense tensor type definition and basic operations
//!
//! This module defines the core `DenseND<T>` type and provides basic creation
//! and accessor methods. Additional operations are organized in separate modules.

use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut, IxDyn};
use scirs2_core::numeric::Num;

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "T: serde::Serialize")))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: serde::Deserialize<'de>"))
)]
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

    /// Convert the tensor to a flat vector in row-major order
    ///
    /// # Returns
    ///
    /// A vector containing all elements in row-major (C-contiguous) order
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let vec = tensor.to_vec();
    /// assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }

    /// Consume the tensor and return the underlying data as a flat vector
    ///
    /// This is more efficient than `to_vec()` when you don't need the tensor anymore.
    ///
    /// # Returns
    ///
    /// A vector containing all elements. If the tensor is contiguous, this is zero-copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let vec = tensor.into_vec();
    /// assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn into_vec(self) -> Vec<T> {
        // Simply collect into vector (ndarray handles optimization internally)
        self.data.into_iter().collect()
    }

    /// Fill the tensor with values produced by a function
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes a multi-dimensional index and returns a value
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::zeros(&[2, 3]);
    /// tensor.fill_with(|idx| (idx[0] + idx[1]) as f64);
    /// assert_eq!(tensor[&[0, 0]], 0.0);
    /// assert_eq!(tensor[&[1, 2]], 3.0);
    /// ```
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut(&[usize]) -> T,
    {
        let shape = self.shape_vec();
        if shape.is_empty() {
            return;
        }

        let mut indices = vec![0; shape.len()];

        for i in 0..self.len() {
            // Convert linear index to multi-dimensional index (row-major order)
            let mut remaining = i;
            for d in (0..shape.len()).rev() {
                indices[d] = remaining % shape[d];
                remaining /= shape[d];
            }

            let value = f(&indices);
            self.data[&indices[..]] = value;
        }
    }

    /// Efficiently copy data from another tensor
    ///
    /// This is more efficient than `clone()` when the tensor already exists.
    ///
    /// # Arguments
    ///
    /// * `source` - The source tensor to copy from
    ///
    /// # Panics
    ///
    /// Panics if the shapes don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let source = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let mut target = DenseND::<f64>::zeros(&[2, 2]);
    /// target.clone_from(&source);
    /// assert_eq!(target[&[0, 0]], 1.0);
    /// assert_eq!(target[&[1, 1]], 4.0);
    /// ```
    pub fn clone_from(&mut self, source: &Self) {
        assert_eq!(
            self.shape(),
            source.shape(),
            "Shape mismatch in clone_from: {:?} vs {:?}",
            self.shape(),
            source.shape()
        );
        self.data.assign(&source.data);
    }

    /// Create an iterator over all elements in row-major order
    ///
    /// # Returns
    ///
    /// An iterator yielding references to all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let sum: f64 = tensor.iter().sum();
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Create a mutable iterator over all elements in row-major order
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let mut tensor = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// for elem in tensor.iter_mut() {
    ///     *elem *= 2.0;
    /// }
    /// assert_eq!(tensor[&[0, 0]], 2.0);
    /// assert_eq!(tensor[&[1, 1]], 8.0);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Get the number of bytes used by the tensor data
    ///
    /// # Returns
    ///
    /// The memory footprint in bytes (approximate)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[100, 100]);
    /// let bytes = tensor.size_bytes();
    /// assert_eq!(bytes, 100 * 100 * std::mem::size_of::<f64>());
    /// ```
    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Check if two tensors have the same shape
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to compare with
    ///
    /// # Returns
    ///
    /// `true` if shapes match, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let a = DenseND::<f64>::zeros(&[2, 3]);
    /// let b = DenseND::<f64>::ones(&[2, 3]);
    /// let c = DenseND::<f64>::zeros(&[3, 2]);
    /// assert!(a.same_shape(&b));
    /// assert!(!a.same_shape(&c));
    /// ```
    pub fn same_shape(&self, other: &Self) -> bool {
        self.shape() == other.shape()
    }
}
