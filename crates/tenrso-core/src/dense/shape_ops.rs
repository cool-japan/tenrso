//! Shape manipulation operations on tensors
//!
//! This module provides comprehensive shape manipulation including reshape, permute,
//! unfold/fold (matricization/tensorization), squeeze/unsqueeze, and axis operations.

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array2, IxDyn};
use scirs2_core::numeric::Num;

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
        if let Ok(reshaped) = self.data.view().into_shape_with_order(IxDyn(new_shape)) {
            Ok(Self {
                data: reshaped.to_owned(),
            })
        } else {
            let flat: Vec<T> = self.data.iter().cloned().collect();
            Self::from_vec(flat, new_shape)
        }
    }

    /// Permute (transpose) the axes of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of axes (must be a permutation of 0..rank)
    ///
    /// # Returns
    ///
    /// A tensor with permuted axes
    ///
    /// # Errors
    ///
    /// Returns an error if `axes` is not a valid permutation.
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
                "Permutation axes length {} does not match tensor rank {}",
                axes.len(),
                self.rank()
            );
        }
        let mut seen = vec![false; self.rank()];
        for &axis in axes {
            if axis >= self.rank() {
                anyhow::bail!("Invalid axis {} for rank {}", axis, self.rank());
            }
            if seen[axis] {
                anyhow::bail!("Duplicate axis {} in permutation", axis);
            }
            seen[axis] = true;
        }
        let permuted = self.data.clone().permuted_axes(IxDyn(axes));
        Ok(Self { data: permuted })
    }

    /// Unfold (matricize) the tensor along a specific mode.
    ///
    /// Mode-n unfolding arranges the mode-n fibers as columns of a matrix.
    /// This is critical for tensor decompositions (CP, Tucker, TT).
    ///
    /// # Arguments
    ///
    /// * `mode` - The mode along which to unfold
    ///
    /// # Returns
    ///
    /// A 2D matrix where mode-n fibers are columns
    ///
    /// # Errors
    ///
    /// Returns an error if mode is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3]
    /// ).unwrap();
    ///
    /// let unfolded = tensor.unfold(0).unwrap();
    /// assert_eq!(unfolded.shape(), &[2, 3]);
    /// ```
    pub fn unfold(&self, mode: usize) -> anyhow::Result<Array2<T>> {
        if mode >= self.rank() {
            anyhow::bail!("Mode {} out of bounds for rank {}", mode, self.rank());
        }

        let shape = self.shape();
        let rows = shape[mode];
        let cols: usize = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != mode)
            .map(|(_, &s)| s)
            .product();

        // Permute so that mode becomes the first axis
        let mut perm: Vec<usize> = vec![mode];
        perm.extend((0..mode).chain((mode + 1)..self.rank()));

        let permuted = self.permute(&perm)?;

        // Reshape to matrix
        let reshaped = permuted.reshape(&[rows, cols])?;

        reshaped
            .data
            .into_dimensionality::<scirs2_core::ndarray_ext::Ix2>()
            .map_err(|e| anyhow::anyhow!("Failed to convert to 2D: {}", e))
    }

    /// Fold (tensorize) a matrix back into a tensor.
    ///
    /// This is the inverse of unfold. Given a matrix and a target shape,
    /// it reconstructs the tensor such that unfold(fold(matrix)) == matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The 2D matrix to fold
    /// * `shape` - The target tensor shape
    /// * `mode` - The mode that was used for unfolding
    ///
    /// # Returns
    ///
    /// A tensor with the specified shape
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are incompatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    /// use scirs2_core::ndarray_ext::Array2;
    ///
    /// let matrix: Array2<f64> = Array2::zeros((2, 6));
    /// let tensor = DenseND::fold(&matrix, &[2, 3, 2], 0).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 3, 2]);
    /// ```
    pub fn fold(matrix: &Array2<T>, shape: &[usize], mode: usize) -> anyhow::Result<Self> {
        if mode >= shape.len() {
            anyhow::bail!("Mode {} out of bounds for target shape {:?}", mode, shape);
        }

        let expected_rows = shape[mode];
        let expected_cols: usize = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != mode)
            .map(|(_, &s)| s)
            .product();

        if matrix.shape()[0] != expected_rows || matrix.shape()[1] != expected_cols {
            anyhow::bail!(
                "Matrix shape {:?} incompatible with target shape {:?} at mode {}",
                matrix.shape(),
                shape,
                mode
            );
        }

        // Create intermediate shape for reshape
        let mut intermediate_shape = vec![shape[mode]];
        for (i, &s) in shape.iter().enumerate() {
            if i != mode {
                intermediate_shape.push(s);
            }
        }

        // Reshape matrix to intermediate tensor
        let flat: Vec<T> = matrix.iter().cloned().collect();
        let intermediate = Self::from_vec(flat, &intermediate_shape)?;

        // Reverse permutation to get original axis order
        let mut inverse_perm = vec![0; shape.len()];
        inverse_perm[mode] = 0;
        let mut idx = 1;
        for (i, perm_val) in inverse_perm.iter_mut().enumerate() {
            if i != mode {
                *perm_val = idx;
                idx += 1;
            }
        }

        intermediate.permute(&inverse_perm)
    }

    /// Remove all singleton dimensions (dimensions of size 1).
    ///
    /// Returns the tensor unchanged when no size-1 axes are present. If every
    /// axis has size 1, the result is a rank-0 (scalar) tensor holding the
    /// single element.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[1, 3, 1, 5, 1]);
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed.shape(), &[3, 5]);
    ///
    /// // All size-1 axes collapse to a 0-D scalar
    /// let scalar = DenseND::<f64>::ones(&[1, 1, 1]).squeeze();
    /// assert_eq!(scalar.shape(), &[] as &[usize]);
    /// assert_eq!(scalar.rank(), 0);
    /// ```
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&s| s != 1).copied().collect();

        // Reshape handles the 0-D case correctly (empty shape, product == 1).
        // Fall back to a clone if (hypothetically) the reshape were to fail;
        // squeeze cannot meaningfully change the total element count.
        self.reshape(&new_shape).unwrap_or_else(|_| self.clone())
    }

    /// Remove a specific singleton dimension.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove (must have size 1)
    ///
    /// # Errors
    ///
    /// Returns an error if `axis` is out of bounds or the axis does not have
    /// size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[3, 1, 5]);
    /// let squeezed = tensor.squeeze_axis(1).unwrap();
    /// assert_eq!(squeezed.shape(), &[3, 5]);
    /// ```
    #[inline]
    pub fn squeeze_axis(&self, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!(
                "squeeze_axis: axis {} out of bounds for tensor of rank {}",
                axis,
                self.rank()
            );
        }

        if self.shape()[axis] != 1 {
            anyhow::bail!(
                "squeeze_axis: cannot squeeze axis {} with size {} (expected size 1)",
                axis,
                self.shape()[axis]
            );
        }

        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .collect();

        self.reshape(&new_shape)
    }

    /// Remove the given set of singleton dimensions simultaneously.
    ///
    /// Every axis listed in `axes` must be in-bounds and have size 1.
    /// Duplicate axes are reported as errors. The relative order of the
    /// remaining axes is preserved.
    ///
    /// # Arguments
    ///
    /// * `axes` - Axes to drop. May be given in any order.
    ///
    /// # Errors
    ///
    /// Returns an error if any axis is out of bounds, has size != 1, or is
    /// specified more than once.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[1, 3, 1, 5]);
    /// let squeezed = tensor.squeeze_axes(&[0, 2]).unwrap();
    /// assert_eq!(squeezed.shape(), &[3, 5]);
    /// ```
    pub fn squeeze_axes(&self, axes: &[usize]) -> anyhow::Result<Self> {
        let rank = self.rank();
        let mut drop = vec![false; rank];
        for &axis in axes {
            if axis >= rank {
                anyhow::bail!(
                    "squeeze_axes: axis {} out of bounds for tensor of rank {}",
                    axis,
                    rank
                );
            }
            if drop[axis] {
                anyhow::bail!("squeeze_axes: duplicate axis {} in axes list", axis);
            }
            if self.shape()[axis] != 1 {
                anyhow::bail!(
                    "squeeze_axes: cannot squeeze axis {} with size {} (expected size 1)",
                    axis,
                    self.shape()[axis]
                );
            }
            drop[axis] = true;
        }

        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| !drop[i])
            .map(|(_, &s)| s)
            .collect();

        self.reshape(&new_shape)
    }

    /// Add a singleton dimension at the specified axis.
    ///
    /// Valid positions are `0..=rank`. Inserting at `rank` appends the new
    /// axis at the end.
    ///
    /// # Arguments
    ///
    /// * `axis` - Position where the new axis will be inserted
    ///
    /// # Errors
    ///
    /// Returns an error if `axis > rank`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[3, 5]);
    /// let unsqueezed = tensor.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[3, 1, 5]);
    /// ```
    #[inline]
    pub fn unsqueeze(&self, axis: usize) -> anyhow::Result<Self> {
        if axis > self.rank() {
            anyhow::bail!(
                "unsqueeze: axis {} out of bounds for result rank {}",
                axis,
                self.rank() + 1
            );
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);

        self.reshape(&new_shape)
    }

    /// Insert multiple singleton dimensions at once.
    ///
    /// Each index in `axes` is interpreted against the *final* shape (rank
    /// `self.rank() + axes.len()`). Duplicate positions are reported as
    /// errors. Indices are applied in sorted ascending order so their meaning
    /// matches the output layout.
    ///
    /// # Arguments
    ///
    /// * `axes` - Positions in the final tensor where size-1 axes should be
    ///   inserted. May be given in any order.
    ///
    /// # Errors
    ///
    /// Returns an error if any position is out of bounds or is duplicated.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3]);
    /// let expanded = tensor.unsqueeze_axes(&[0, 2]).unwrap();
    /// assert_eq!(expanded.shape(), &[1, 2, 1, 3]);
    /// ```
    pub fn unsqueeze_axes(&self, axes: &[usize]) -> anyhow::Result<Self> {
        let final_rank = self.rank() + axes.len();

        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_unstable();

        // Validate bounds and detect duplicates after sorting.
        let mut prev: Option<usize> = None;
        for &axis in &sorted_axes {
            if axis >= final_rank {
                anyhow::bail!(
                    "unsqueeze_axes: axis {} out of bounds for result rank {}",
                    axis,
                    final_rank
                );
            }
            if Some(axis) == prev {
                anyhow::bail!("unsqueeze_axes: duplicate axis {} in axes list", axis);
            }
            prev = Some(axis);
        }

        // Build the final shape by starting with the existing shape and
        // inserting a 1 at each requested position (ascending) in order.
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        for axis in sorted_axes {
            new_shape.insert(axis, 1);
        }

        self.reshape(&new_shape)
    }

    /// Flatten tensor to 1D
    ///
    /// Returns a 1D view of the tensor in row-major (C) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let flat = tensor.flatten();
    ///
    /// assert_eq!(flat.shape(), &[6]);
    /// assert_eq!(flat[&[0]], 1.0);
    /// assert_eq!(flat[&[5]], 6.0);
    /// ```
    pub fn flatten(&self) -> Self {
        let total = self.len();
        let flat = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&[total]))
            .expect("Flatten should always succeed");
        Self { data: flat }
    }

    /// Alias for flatten (returns a 1D view in row-major order)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let raveled = tensor.ravel();
    /// assert_eq!(raveled.shape(), &[4]);
    /// ```
    pub fn ravel(&self) -> Self {
        self.flatten()
    }

    /// Swap two axes of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let swapped = tensor.swapaxes(0, 2).unwrap();
    /// assert_eq!(swapped.shape(), &[4, 3, 2]);
    /// ```
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> anyhow::Result<Self> {
        if axis1 >= self.rank() || axis2 >= self.rank() {
            anyhow::bail!(
                "Axes {} and {} out of bounds for rank {}",
                axis1,
                axis2,
                self.rank()
            );
        }

        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(axis1, axis2);
        self.permute(&perm)
    }

    /// Move an axis to a new position
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4, 5]);
    /// let moved = tensor.moveaxis(3, 1).unwrap();
    /// assert_eq!(moved.shape(), &[2, 5, 3, 4]);
    /// ```
    pub fn moveaxis(&self, source: usize, destination: usize) -> anyhow::Result<Self> {
        if source >= self.rank() || destination >= self.rank() {
            anyhow::bail!(
                "Source {} or destination {} out of bounds for rank {}",
                source,
                destination,
                self.rank()
            );
        }

        let mut perm: Vec<usize> = (0..self.rank()).collect();
        let axis = perm.remove(source);
        perm.insert(destination, axis);
        self.permute(&perm)
    }

    /// View input as array with at least one dimension.
    ///
    /// Scalar inputs (rank 0) are converted to 1D arrays.
    /// Higher-rank inputs are returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // 1D input stays 1D
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let result = tensor.atleast_1d();
    /// assert_eq!(result.shape(), &[3]);
    ///
    /// // Scalar becomes 1D
    /// let scalar = DenseND::<f64>::from_elem(&[], 5.0);
    /// let result = scalar.atleast_1d();
    /// assert_eq!(result.rank(), 1);
    /// ```
    pub fn atleast_1d(&self) -> Self {
        if self.rank() == 0 {
            // Convert scalar to 1D array: a rank-0 tensor has exactly 1
            // element, so reshaping to `[1]` preserves the element count.
            self.reshape(&[1])
                .expect("atleast_1d: scalar has exactly one element")
        } else {
            self.clone()
        }
    }

    /// View input as array with at least two dimensions.
    ///
    /// Inputs with rank < 2 are converted to 2D arrays.
    /// Higher-rank inputs are returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // 2D input stays 2D
    /// let tensor = DenseND::<f64>::zeros(&[2, 3]);
    /// let result = tensor.atleast_2d();
    /// assert_eq!(result.shape(), &[2, 3]);
    ///
    /// // 1D becomes 2D (1, N)
    /// let vec = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let result = vec.atleast_2d();
    /// assert_eq!(result.shape(), &[1, 3]);
    /// ```
    pub fn atleast_2d(&self) -> Self {
        // Each branch below preserves the total number of elements, so
        // `reshape` cannot fail. `.expect` documents that invariant.
        match self.rank() {
            0 => self
                .reshape(&[1, 1])
                .expect("atleast_2d: scalar has exactly one element"),
            1 => {
                let n = self.shape()[0];
                self.reshape(&[1, n])
                    .expect("atleast_2d: 1×n preserves 1D element count")
            }
            _ => self.clone(),
        }
    }

    /// View input as array with at least three dimensions.
    ///
    /// Inputs with rank < 3 are converted to 3D arrays.
    /// Higher-rank inputs are returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // 3D input stays 3D
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let result = tensor.atleast_3d();
    /// assert_eq!(result.shape(), &[2, 3, 4]);
    ///
    /// // 1D becomes 3D (1, N, 1)
    /// let vec = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let result = vec.atleast_3d();
    /// assert_eq!(result.shape(), &[1, 3, 1]);
    ///
    /// // 2D becomes 3D (M, N, 1)
    /// let mat = DenseND::<f64>::zeros(&[2, 3]);
    /// let result = mat.atleast_3d();
    /// assert_eq!(result.shape(), &[2, 3, 1]);
    /// ```
    pub fn atleast_3d(&self) -> Self {
        // Each reshape below preserves element count; any failure would be
        // an internal logic bug.
        match self.rank() {
            0 => self
                .reshape(&[1, 1, 1])
                .expect("atleast_3d: scalar has exactly one element"),
            1 => {
                let n = self.shape()[0];
                self.reshape(&[1, n, 1])
                    .expect("atleast_3d: 1×n×1 preserves 1D element count")
            }
            2 => {
                let m = self.shape()[0];
                let n = self.shape()[1];
                self.reshape(&[m, n, 1])
                    .expect("atleast_3d: m×n×1 preserves 2D element count")
            }
            _ => self.clone(),
        }
    }

    /// Expand dimensions by inserting a new axis (alias for unsqueeze).
    ///
    /// This is a convenience method that's more explicit about intent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3]);
    /// let expanded = tensor.expand_dims(1).unwrap();
    /// assert_eq!(expanded.shape(), &[2, 1, 3]);
    /// ```
    pub fn expand_dims(&self, axis: usize) -> anyhow::Result<Self> {
        self.unsqueeze(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    // -- squeeze --------------------------------------------------------

    #[test]
    fn test_squeeze_no_singletons_is_noop_with_data() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3]);
        assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_squeeze_single_interior_axis() {
        // Shape [3, 1, 4] -> [3, 4] with element order preserved.
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let tensor = DenseND::<f64>::from_vec(data.clone(), &[3, 1, 4]).unwrap();
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[3, 4]);
        assert_eq!(squeezed.to_vec(), data);
    }

    #[test]
    fn test_squeeze_multiple_axes() {
        let tensor = DenseND::<f64>::zeros(&[1, 2, 1, 3, 1]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3]);
        assert_eq!(squeezed.rank(), 2);
    }

    #[test]
    fn test_squeeze_to_scalar() {
        // All axes of size 1 collapse to a rank-0 tensor with 1 element.
        let tensor = DenseND::<f64>::from_elem(&[1, 1, 1], 42.0);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[] as &[usize]);
        assert_eq!(squeezed.rank(), 0);
        assert_eq!(squeezed.len(), 1);
        let iter_val = *squeezed.iter().next().expect("scalar has one element");
        assert_eq!(iter_val, 42.0);
    }

    #[test]
    fn test_squeeze_preserves_data_values() {
        let src = array![[[1.0_f64, 2.0, 3.0]]]; // shape [1, 1, 3]
        let tensor = DenseND::from_array(src.into_dyn());
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[3]);
        assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    // -- squeeze_axis ---------------------------------------------------

    #[test]
    fn test_squeeze_axis_success() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let tensor = DenseND::<f64>::from_vec(data.clone(), &[3, 1, 4]).unwrap();
        let squeezed = tensor.squeeze_axis(1).unwrap();
        assert_eq!(squeezed.shape(), &[3, 4]);
        assert_eq!(squeezed.to_vec(), data);
    }

    #[test]
    fn test_squeeze_axis_out_of_bounds() {
        let tensor = DenseND::<f64>::zeros(&[2, 1, 3]);
        let err = tensor.squeeze_axis(3).expect_err("axis out of bounds");
        let msg = format!("{err}");
        assert!(
            msg.contains("out of bounds"),
            "expected OOB error, got: {msg}"
        );
    }

    #[test]
    fn test_squeeze_axis_not_size_one() {
        let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
        let err = tensor.squeeze_axis(1).expect_err("axis not size 1");
        let msg = format!("{err}");
        assert!(
            msg.contains("expected size 1"),
            "expected size-1 error, got: {msg}"
        );
    }

    #[test]
    fn test_squeeze_axis_rejects_rank_equal_index() {
        // axis == rank() is out of bounds for squeeze_axis.
        let tensor = DenseND::<f64>::zeros(&[1, 1]);
        assert!(tensor.squeeze_axis(2).is_err());
    }

    // -- squeeze_axes ---------------------------------------------------

    #[test]
    fn test_squeeze_axes_multiple() {
        let tensor = DenseND::<f64>::from_elem(&[1, 3, 1, 5], 7.0);
        let squeezed = tensor.squeeze_axes(&[0, 2]).unwrap();
        assert_eq!(squeezed.shape(), &[3, 5]);
        assert_eq!(squeezed.rank(), 2);
        // Data values preserved.
        assert!(squeezed.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_squeeze_axes_unordered_input() {
        let tensor = DenseND::<f64>::zeros(&[1, 2, 1, 3, 1]);
        // Pass indices out of order; function must handle internally.
        let squeezed = tensor.squeeze_axes(&[4, 0, 2]).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3]);
    }

    #[test]
    fn test_squeeze_axes_empty_is_noop() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let squeezed = tensor.squeeze_axes(&[]).unwrap();
        assert_eq!(squeezed.shape(), &[2, 2]);
        assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_squeeze_axes_rejects_non_singleton() {
        let tensor = DenseND::<f64>::zeros(&[1, 2, 1]);
        assert!(tensor.squeeze_axes(&[0, 1]).is_err());
    }

    #[test]
    fn test_squeeze_axes_rejects_duplicate() {
        let tensor = DenseND::<f64>::zeros(&[1, 2, 1]);
        let err = tensor
            .squeeze_axes(&[0, 0])
            .expect_err("duplicate axis must error");
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "got: {msg}");
    }

    #[test]
    fn test_squeeze_axes_rejects_out_of_bounds() {
        let tensor = DenseND::<f64>::zeros(&[1, 2, 1]);
        assert!(tensor.squeeze_axes(&[5]).is_err());
    }

    // -- unsqueeze ------------------------------------------------------

    #[test]
    fn test_unsqueeze_at_front_preserves_data() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = DenseND::<f64>::from_vec(data.clone(), &[2, 3]).unwrap();
        let expanded = tensor.unsqueeze(0).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 3]);
        assert_eq!(expanded.to_vec(), data);
    }

    #[test]
    fn test_unsqueeze_at_end_equals_rank() {
        // Inserting at `rank()` appends a trailing size-1 axis.
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let expanded = tensor.unsqueeze(2).unwrap();
        assert_eq!(expanded.shape(), &[2, 2, 1]);
        assert_eq!(expanded.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_unsqueeze_out_of_bounds() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        // axis > rank() is invalid.
        let err = tensor
            .unsqueeze(3)
            .expect_err("axis beyond rank must error");
        let msg = format!("{err}");
        assert!(msg.contains("out of bounds"), "got: {msg}");
    }

    #[test]
    fn test_unsqueeze_on_scalar_creates_1d() {
        let scalar = DenseND::<f64>::from_elem(&[], 9.0);
        let expanded = scalar.unsqueeze(0).unwrap();
        assert_eq!(expanded.shape(), &[1]);
        assert_eq!(expanded[&[0]], 9.0);
    }

    // -- unsqueeze_axes -------------------------------------------------

    #[test]
    fn test_unsqueeze_axes_multiple_positions() {
        let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        // Final rank = 2 + 2 = 4. Insert size-1 axes at positions 0 and 2.
        let expanded = tensor.unsqueeze_axes(&[0, 2]).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 1, 3]);
        // Data values unchanged.
        assert_eq!(expanded.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unsqueeze_axes_unsorted_input() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        // Same as [0, 2] since the implementation sorts internally.
        let expanded = tensor.unsqueeze_axes(&[2, 0]).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 1, 3]);
    }

    #[test]
    fn test_unsqueeze_axes_empty_is_noop() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        let expanded = tensor.unsqueeze_axes(&[]).unwrap();
        assert_eq!(expanded.shape(), &[2, 3]);
    }

    #[test]
    fn test_unsqueeze_axes_rejects_out_of_bounds() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        // final_rank = 3, so axis 5 is invalid.
        assert!(tensor.unsqueeze_axes(&[5]).is_err());
    }

    #[test]
    fn test_unsqueeze_axes_rejects_duplicate_positions() {
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        let err = tensor
            .unsqueeze_axes(&[1, 1])
            .expect_err("duplicate positions must error");
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "got: {msg}");
    }

    #[test]
    fn test_unsqueeze_axes_append_at_final_boundary() {
        // Valid boundary: largest index is final_rank - 1. Here final_rank = 3,
        // so position 2 appends the new size-1 axis at the end.
        let tensor = DenseND::<f64>::zeros(&[2, 3]);
        let expanded = tensor.unsqueeze_axes(&[2]).unwrap();
        assert_eq!(expanded.shape(), &[2, 3, 1]);
        // And index >= final_rank is rejected.
        let tensor2 = DenseND::<f64>::zeros(&[2, 3]);
        assert!(tensor2.unsqueeze_axes(&[3]).is_err());
    }

    // -- round-trips ----------------------------------------------------

    #[test]
    fn test_unsqueeze_then_squeeze_returns_original_shape() {
        let original =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let expanded = original.unsqueeze(1).unwrap();
        assert_eq!(expanded.shape(), &[2, 1, 3]);
        let restored = expanded.squeeze();
        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.to_vec(), original.to_vec());
    }

    #[test]
    fn test_unsqueeze_axes_then_squeeze_axes_roundtrip() {
        let original =
            DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let expanded = original.unsqueeze_axes(&[0, 2]).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 1, 3]);
        let restored = expanded.squeeze_axes(&[0, 2]).unwrap();
        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.to_vec(), original.to_vec());
    }
}
