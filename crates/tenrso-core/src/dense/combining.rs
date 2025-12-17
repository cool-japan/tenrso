//! Tensor combining and splitting operations
//!
//! This module provides operations for combining multiple tensors (concatenate, stack)
//! and splitting tensors (split, chunk).

use super::types::DenseND;
use scirs2_core::ndarray_ext::Axis;
use scirs2_core::numeric::Num;

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Concatenate multiple tensors along an existing axis.
    ///
    /// All tensors must have the same shape except along the concatenation axis.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to concatenate
    /// * `axis` - Axis along which to concatenate
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements across all tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor list is empty
    /// - Shapes are incompatible
    /// - Axis is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::ones(&[2, 3]);
    /// let b = DenseND::<f64>::zeros(&[2, 3]);
    ///
    /// let concatenated = DenseND::concatenate(&[a, b], 0).unwrap();
    /// assert_eq!(concatenated.shape(), &[4, 3]);
    /// ```
    pub fn concatenate(tensors: &[Self], axis: usize) -> anyhow::Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot concatenate empty tensor list");
        }

        let rank = tensors[0].rank();
        if axis >= rank {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, rank);
        }

        // Verify all tensors have compatible shapes
        let reference_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.rank() != rank {
                anyhow::bail!("Tensor {} has rank {}, expected {}", i, tensor.rank(), rank);
            }
            for (dim, (&s1, &s2)) in reference_shape
                .iter()
                .zip(tensor.shape().iter())
                .enumerate()
            {
                if dim != axis && s1 != s2 {
                    anyhow::bail!("Shape mismatch at dimension {}: {} vs {}", dim, s1, s2);
                }
            }
        }

        // Compute result shape
        let mut result_shape = reference_shape.to_vec();
        result_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

        // Create views and concatenate using ndarray
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let concatenated = scirs2_core::ndarray::concatenate(Axis(axis), &views)?;

        Ok(Self { data: concatenated })
    }

    /// Stack multiple tensors along a new axis.
    ///
    /// All tensors must have the same shape. The result will have rank+1 dimensions.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to stack
    /// * `axis` - Position where the new axis will be inserted
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor list is empty
    /// - Shapes don't match
    /// - Axis is out of bounds for the result
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let a = DenseND::<f64>::ones(&[2, 3]);
    /// let b = DenseND::<f64>::zeros(&[2, 3]);
    ///
    /// let stacked = DenseND::stack(&[a, b], 0).unwrap();
    /// assert_eq!(stacked.shape(), &[2, 2, 3]);
    /// ```
    pub fn stack(tensors: &[Self], axis: usize) -> anyhow::Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot stack empty tensor list");
        }

        let rank = tensors[0].rank();
        if axis > rank {
            anyhow::bail!("Axis {} out of bounds for result rank {}", axis, rank + 1);
        }

        // Verify all tensors have the same shape
        let reference_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != reference_shape {
                anyhow::bail!(
                    "Tensor {} has shape {:?}, expected {:?}",
                    i,
                    tensor.shape(),
                    reference_shape
                );
            }
        }

        // Stack by first unsqueezing each tensor, then concatenating
        let unsqueezed: Result<Vec<_>, _> = tensors.iter().map(|t| t.unsqueeze(axis)).collect();
        let unsqueezed = unsqueezed?;

        Self::concatenate(&unsqueezed, axis)
    }

    /// Split a tensor into multiple sub-tensors along an axis.
    ///
    /// The tensor is split into `num_splits` equal parts along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `num_splits` - Number of splits (must evenly divide the axis size)
    /// * `axis` - Axis along which to split
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Axis is out of bounds
    /// - The axis size is not evenly divisible by num_splits
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[4, 3]);
    /// let parts = tensor.split(2, 0).unwrap();
    ///
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(parts[0].shape(), &[2, 3]);
    /// assert_eq!(parts[1].shape(), &[2, 3]);
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

        let split_size = axis_size / num_splits;
        let mut result = Vec::with_capacity(num_splits);

        for i in 0..num_splits {
            let start = i * split_size;
            let end = (i + 1) * split_size;

            let indices: Vec<usize> = (start..end).collect();
            let part = self.select_indices(&indices, axis)?;
            result.push(part);
        }

        Ok(result)
    }

    /// Chunk a tensor into multiple sub-tensors of a given size.
    ///
    /// The last chunk may be smaller if the axis size is not evenly divisible.
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Size of each chunk along the axis
    /// * `axis` - Axis along which to chunk
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of elements
    ///
    /// # Errors
    ///
    /// Returns an error if axis is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::ones(&[5, 3]);
    /// let chunks = tensor.chunk(2, 0).unwrap();
    ///
    /// assert_eq!(chunks.len(), 3);
    /// assert_eq!(chunks[0].shape(), &[2, 3]);
    /// assert_eq!(chunks[1].shape(), &[2, 3]);
    /// assert_eq!(chunks[2].shape(), &[1, 3]); // Last chunk is smaller
    /// ```
    pub fn chunk(&self, chunk_size: usize, axis: usize) -> anyhow::Result<Vec<Self>> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if chunk_size == 0 {
            anyhow::bail!("Chunk size must be greater than 0");
        }

        let axis_size = self.shape()[axis];
        let num_chunks = axis_size.div_ceil(chunk_size);
        let mut result = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = std::cmp::min((i + 1) * chunk_size, axis_size);

            let indices: Vec<usize> = (start..end).collect();
            let chunk = self.select_indices(&indices, axis)?;
            result.push(chunk);
        }

        Ok(result)
    }
}
