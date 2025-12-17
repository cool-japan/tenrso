//! Indexing and selection operations on tensors
//!
//! This module provides various indexing and selection methods including
//! direct element access, boolean mask selection, and index array selection.

use super::types::DenseND;
use scirs2_core::numeric::Num;

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Get an element by index without panicking
    ///
    /// # Arguments
    ///
    /// * `index` - Multi-dimensional index
    ///
    /// # Returns
    ///
    /// Some reference to the element if the index is valid, None otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// assert_eq!(tensor.get(&[0, 1]), Some(&2.0));
    /// assert_eq!(tensor.get(&[5, 5]), None);
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        if index.len() != self.rank() {
            return None;
        }
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape()[i] {
                return None;
            }
        }
        self.data.get(index)
    }

    /// Get a mutable reference to an element by index without panicking
    ///
    /// # Arguments
    ///
    /// * `index` - Multi-dimensional index
    ///
    /// # Returns
    ///
    /// Some mutable reference to the element if the index is valid, None otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// if let Some(elem) = tensor.get_mut(&[0, 1]) {
    ///     *elem = 10.0;
    /// }
    /// assert_eq!(tensor[&[0, 1]], 10.0);
    /// ```
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        if index.len() != self.rank() {
            return None;
        }
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape()[i] {
                return None;
            }
        }
        self.data.get_mut(index)
    }

    /// Get the underlying data as a slice if the tensor is contiguous
    ///
    /// # Returns
    ///
    /// A slice of the data if contiguous, panics otherwise
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous in memory
    ///
    /// # Note
    ///
    /// For a non-panicking version, use [`try_as_slice`](Self::try_as_slice)
    pub fn as_slice(&self) -> &[T] {
        self.data
            .as_slice()
            .expect("Tensor is not contiguous in memory")
    }

    /// Try to get the underlying data as a slice if the tensor is contiguous
    ///
    /// This is a safe, non-panicking version of [`as_slice`](Self::as_slice).
    ///
    /// # Returns
    ///
    /// - `Some(&[T])` if the tensor is contiguous in memory
    /// - `None` if the tensor is not contiguous
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// assert!(tensor.try_as_slice().is_some());
    ///
    /// // After transpose, may not be contiguous
    /// let transposed = tensor.permute(&[1, 0]).unwrap();
    /// // try_as_slice won't panic even if not contiguous
    /// let _ = transposed.try_as_slice();
    /// ```
    pub fn try_as_slice(&self) -> Option<&[T]> {
        self.data.as_slice()
    }

    /// Try to get a mutable slice of the underlying data if the tensor is contiguous
    ///
    /// This is a safe, non-panicking mutable version of slice access.
    ///
    /// # Returns
    ///
    /// - `Some(&mut [T])` if the tensor is contiguous in memory
    /// - `None` if the tensor is not contiguous
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::zeros(&[4]);
    /// if let Some(slice) = tensor.try_as_slice_mut() {
    ///     slice[0] = 1.0;
    ///     slice[1] = 2.0;
    /// }
    /// assert_eq!(tensor[&[0]], 1.0);
    /// assert_eq!(tensor[&[1]], 2.0);
    /// ```
    pub fn try_as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.data.as_slice_mut()
    }

    /// Get element with detailed error reporting on out-of-bounds access
    ///
    /// Similar to `get()` but returns a Result with detailed error messages
    /// instead of Option, making it easier to debug indexing issues.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// assert_eq!(*tensor.get_checked(&[0, 1]).unwrap(), 2.0);
    ///
    /// // Out of bounds gives clear error
    /// assert!(tensor.get_checked(&[5, 5]).is_err());
    /// ```
    pub fn get_checked(&self, index: &[usize]) -> anyhow::Result<&T> {
        if index.len() != self.rank() {
            anyhow::bail!(
                "Index has {} dimensions but tensor has rank {}",
                index.len(),
                self.rank()
            );
        }

        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape()[i] {
                anyhow::bail!(
                    "Index {} is out of bounds for dimension {} with size {}",
                    idx,
                    i,
                    self.shape()[i]
                );
            }
        }

        Ok(self
            .data
            .get(index)
            .expect("Index validation passed but get failed"))
    }

    /// Get mutable element with detailed error reporting
    ///
    /// Similar to `get_mut()` but returns a Result with detailed error messages.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::zeros(&[2, 2]);
    /// *tensor.get_checked_mut(&[0, 1]).unwrap() = 5.0;
    /// assert_eq!(tensor[&[0, 1]], 5.0);
    /// ```
    pub fn get_checked_mut(&mut self, index: &[usize]) -> anyhow::Result<&mut T> {
        if index.len() != self.rank() {
            anyhow::bail!(
                "Index has {} dimensions but tensor has rank {}",
                index.len(),
                self.rank()
            );
        }

        let rank = self.rank();
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape()[i] {
                anyhow::bail!(
                    "Index {} is out of bounds for dimension {} with size {}",
                    idx,
                    i,
                    self.shape()[i]
                );
            }
        }

        // Store shape for error message before mut borrow
        let shape = self.shape().to_vec();

        self.data.get_mut(index).ok_or_else(|| {
            anyhow::anyhow!(
                "Index validation passed but get_mut failed for rank {} tensor with shape {:?}",
                rank,
                shape
            )
        })
    }

    /// Get first element of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(*tensor.first().unwrap(), 1.0);
    /// ```
    pub fn first(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.data.iter().next()
        }
    }

    /// Get last element of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(*tensor.last().unwrap(), 3.0);
    /// ```
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.data.iter().last()
        }
    }

    /// Select elements using a boolean mask
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean array with same length as flattened tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let mask = vec![true, false, true, false];
    /// let selected = tensor.select_mask(&mask).unwrap();
    ///
    /// assert_eq!(selected.shape(), &[2]);
    /// assert_eq!(selected[&[0]], 1.0);
    /// assert_eq!(selected[&[1]], 3.0);
    /// ```
    pub fn select_mask(&self, mask: &[bool]) -> anyhow::Result<Self> {
        if mask.len() != self.len() {
            anyhow::bail!(
                "Mask length {} does not match tensor size {}",
                mask.len(),
                self.len()
            );
        }

        let selected: Vec<T> = self
            .data
            .iter()
            .zip(mask.iter())
            .filter_map(|(val, &keep)| if keep { Some(val.clone()) } else { None })
            .collect();

        if selected.is_empty() {
            anyhow::bail!("Mask selected no elements");
        }

        let len = selected.len();
        Self::from_vec(selected, &[len])
    }

    /// Select elements along an axis using an index array
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of indices to select
    /// * `axis` - Axis along which to select
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    /// let selected = tensor.select_indices(&[0, 2], 0).unwrap();
    ///
    /// assert_eq!(selected.shape(), &[2, 2]);
    /// assert_eq!(selected[&[0, 0]], 1.0);
    /// assert_eq!(selected[&[1, 0]], 5.0);
    /// ```
    pub fn select_indices(&self, indices: &[usize], axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        for &idx in indices {
            if idx >= self.shape()[axis] {
                anyhow::bail!(
                    "Index {} out of bounds for axis {} with size {}",
                    idx,
                    axis,
                    self.shape()[axis]
                );
            }
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = indices.len();

        let mut result = Self::zeros(&new_shape);

        for (i, &src_idx) in indices.iter().enumerate() {
            let src_indices = super::functions::generate_indices(self.shape(), axis, src_idx);
            let dst_indices = super::functions::generate_indices(&new_shape, axis, i);

            for (src, dst) in src_indices.iter().zip(dst_indices.iter()) {
                result[dst.as_slice()] = self[src.as_slice()].clone();
            }
        }

        Ok(result)
    }

    /// Filter elements using a predicate function
    ///
    /// Returns a 1D tensor containing only elements that satisfy the predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let filtered = tensor.filter(|&x| x > 2.5).unwrap();
    ///
    /// assert_eq!(filtered.len(), 3);
    /// assert_eq!(filtered[&[0]], 3.0);
    /// ```
    pub fn filter<F>(&self, predicate: F) -> anyhow::Result<Self>
    where
        F: Fn(&T) -> bool,
    {
        let filtered: Vec<T> = self.data.iter().filter(|x| predicate(x)).cloned().collect();

        if filtered.is_empty() {
            anyhow::bail!("Filter resulted in empty tensor");
        }

        let len = filtered.len();
        Self::from_vec(filtered, &[len])
    }

    /// Take elements from an array along an axis using indices.
    ///
    /// This is a more general version of `select_indices` that allows duplicate indices
    /// and negative indexing (wrapping around).
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of indices to take. Can contain duplicates.
    /// * `axis` - Axis along which to take elements.
    ///
    /// # Complexity
    ///
    /// O(n * m) where n is the number of indices and m is the size of other dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// // Take with duplicates
    /// let taken = tensor.take(&[0, 2, 2, 1], 0).unwrap();
    /// assert_eq!(taken.shape(), &[4]);
    /// assert_eq!(taken[&[0]], 1.0);
    /// assert_eq!(taken[&[1]], 3.0);
    /// assert_eq!(taken[&[2]], 3.0);
    /// assert_eq!(taken[&[3]], 2.0);
    /// ```
    pub fn take(&self, indices: &[usize], axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let axis_size = self.shape()[axis];

        // Validate all indices
        for &idx in indices {
            if idx >= axis_size {
                anyhow::bail!(
                    "Index {} out of bounds for axis {} with size {}",
                    idx,
                    axis,
                    axis_size
                );
            }
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = indices.len();

        let mut result = Self::zeros(&new_shape);

        for (i, &src_idx) in indices.iter().enumerate() {
            let src_indices = super::functions::generate_indices(self.shape(), axis, src_idx);
            let dst_indices = super::functions::generate_indices(&new_shape, axis, i);

            for (src, dst) in src_indices.iter().zip(dst_indices.iter()) {
                result[dst.as_slice()] = self[src.as_slice()].clone();
            }
        }

        Ok(result)
    }

    /// Put values into specific positions along an axis.
    ///
    /// Replaces elements at specified indices with corresponding values.
    /// This is an in-place operation.
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of indices where to put values
    /// * `values` - Tensor of values to put (must match shape except at specified axis)
    /// * `axis` - Axis along which to put values
    ///
    /// # Complexity
    ///
    /// O(n * m) where n is the number of indices and m is the size of other dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let mut tensor = DenseND::<f64>::zeros(&[5]);
    /// let values = DenseND::<f64>::from_vec(vec![10.0, 20.0], &[2]).unwrap();
    /// tensor.put(&[1, 3], &values, 0).unwrap();
    ///
    /// assert_eq!(tensor[&[0]], 0.0);
    /// assert_eq!(tensor[&[1]], 10.0);
    /// assert_eq!(tensor[&[2]], 0.0);
    /// assert_eq!(tensor[&[3]], 20.0);
    /// assert_eq!(tensor[&[4]], 0.0);
    /// ```
    pub fn put(&mut self, indices: &[usize], values: &Self, axis: usize) -> anyhow::Result<()> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if values.rank() != self.rank() {
            anyhow::bail!(
                "Values rank {} does not match tensor rank {}",
                values.rank(),
                self.rank()
            );
        }

        // Check that shapes match except at the specified axis
        for i in 0..self.rank() {
            if i != axis && values.shape()[i] != self.shape()[i] {
                anyhow::bail!(
                    "Values shape {:?} incompatible with tensor shape {:?} at axis {}",
                    values.shape(),
                    self.shape(),
                    i
                );
            }
        }

        if values.shape()[axis] != indices.len() {
            anyhow::bail!(
                "Number of indices {} does not match values size {} at axis {}",
                indices.len(),
                values.shape()[axis],
                axis
            );
        }

        let axis_size = self.shape()[axis];

        // Validate and perform put operation
        for (i, &dst_idx) in indices.iter().enumerate() {
            if dst_idx >= axis_size {
                anyhow::bail!(
                    "Index {} out of bounds for axis {} with size {}",
                    dst_idx,
                    axis,
                    axis_size
                );
            }

            let src_indices = super::functions::generate_indices(values.shape(), axis, i);
            let dst_indices = super::functions::generate_indices(self.shape(), axis, dst_idx);

            for (src, dst) in src_indices.iter().zip(dst_indices.iter()) {
                self[dst.as_slice()] = values[src.as_slice()].clone();
            }
        }

        Ok(())
    }

    /// Select entries along an axis using a boolean condition.
    ///
    /// Similar to `select_mask` but works along a specific axis rather than flattening.
    ///
    /// # Arguments
    ///
    /// * `condition` - Boolean array with length equal to axis size
    /// * `axis` - Axis along which to compress
    ///
    /// # Complexity
    ///
    /// O(n * m) where n is the number of True values and m is the size of other dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[3, 2]
    /// ).unwrap();
    ///
    /// // Select rows 0 and 2
    /// let compressed = tensor.compress(&[true, false, true], 0).unwrap();
    /// assert_eq!(compressed.shape(), &[2, 2]);
    /// assert_eq!(compressed[&[0, 0]], 1.0);
    /// assert_eq!(compressed[&[0, 1]], 2.0);
    /// assert_eq!(compressed[&[1, 0]], 5.0);
    /// assert_eq!(compressed[&[1, 1]], 6.0);
    /// ```
    pub fn compress(&self, condition: &[bool], axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if condition.len() != self.shape()[axis] {
            anyhow::bail!(
                "Condition length {} does not match axis {} size {}",
                condition.len(),
                axis,
                self.shape()[axis]
            );
        }

        // Collect indices where condition is true
        let selected_indices: Vec<usize> = condition
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        if selected_indices.is_empty() {
            anyhow::bail!("Condition selected no elements along axis {}", axis);
        }

        // Use take to get the selected elements
        self.take(&selected_indices, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_as_slice_contiguous() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(tensor.try_as_slice().is_some());
        assert_eq!(tensor.try_as_slice().unwrap().len(), 4);
    }

    #[test]
    fn test_try_as_slice_mut_contiguous() {
        let mut tensor = DenseND::<f64>::zeros(&[4]);
        assert!(tensor.try_as_slice_mut().is_some());

        if let Some(slice) = tensor.try_as_slice_mut() {
            slice[0] = 10.0;
            slice[3] = 40.0;
        }

        assert_eq!(tensor[&[0]], 10.0);
        assert_eq!(tensor[&[3]], 40.0);
    }

    #[test]
    fn test_get_checked_success() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(*tensor.get_checked(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get_checked(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*tensor.get_checked(&[1, 0]).unwrap(), 3.0);
        assert_eq!(*tensor.get_checked(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_get_checked_out_of_bounds() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(tensor.get_checked(&[5, 0]).is_err());
        assert!(tensor.get_checked(&[0, 5]).is_err());
        assert!(tensor.get_checked(&[2, 2]).is_err());
    }

    #[test]
    fn test_get_checked_wrong_rank() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(tensor.get_checked(&[0]).is_err());
        assert!(tensor.get_checked(&[0, 0, 0]).is_err());
    }

    #[test]
    fn test_get_checked_mut() {
        let mut tensor = DenseND::<f64>::zeros(&[2, 2]);
        *tensor.get_checked_mut(&[0, 1]).unwrap() = 5.0;
        *tensor.get_checked_mut(&[1, 0]).unwrap() = 10.0;

        assert_eq!(tensor[&[0, 1]], 5.0);
        assert_eq!(tensor[&[1, 0]], 10.0);
    }

    #[test]
    fn test_get_checked_mut_out_of_bounds() {
        let mut tensor = DenseND::<f64>::zeros(&[2, 2]);
        assert!(tensor.get_checked_mut(&[5, 0]).is_err());
        assert!(tensor.get_checked_mut(&[0, 5]).is_err());
    }

    #[test]
    fn test_first() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(*tensor.first().unwrap(), 1.0);
    }

    #[test]
    fn test_first_empty() {
        let tensor = DenseND::<f64>::zeros(&[0]);
        assert!(tensor.first().is_none());
    }

    #[test]
    fn test_last() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(*tensor.last().unwrap(), 3.0);
    }

    #[test]
    fn test_last_empty() {
        let tensor = DenseND::<f64>::zeros(&[0]);
        assert!(tensor.last().is_none());
    }

    #[test]
    fn test_get_vs_get_checked() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Both should succeed for valid index
        assert_eq!(tensor.get(&[0, 1]), Some(&2.0));
        assert_eq!(*tensor.get_checked(&[0, 1]).unwrap(), 2.0);

        // Both should fail for invalid index (different error types)
        assert!(tensor.get(&[5, 5]).is_none());
        assert!(tensor.get_checked(&[5, 5]).is_err());
    }

    #[test]
    fn test_safety_methods_on_multidimensional() {
        let tensor = DenseND::<f64>::ones(&[3, 4, 5]);

        // Test first and last
        assert_eq!(*tensor.first().unwrap(), 1.0);
        assert_eq!(*tensor.last().unwrap(), 1.0);

        // Test get_checked with 3D index
        assert_eq!(*tensor.get_checked(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get_checked(&[2, 3, 4]).unwrap(), 1.0);

        // Test out of bounds
        assert!(tensor.get_checked(&[3, 0, 0]).is_err());
        assert!(tensor.get_checked(&[0, 4, 0]).is_err());
        assert!(tensor.get_checked(&[0, 0, 5]).is_err());
    }
}
