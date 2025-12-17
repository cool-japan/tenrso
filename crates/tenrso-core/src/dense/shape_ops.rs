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
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[1, 3, 1, 5, 1]);
    /// let squeezed = tensor.squeeze();
    /// assert_eq!(squeezed.shape(), &[3, 5]);
    /// ```
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&s| s != 1).copied().collect();

        if new_shape.is_empty() {
            // If all dimensions were 1, result is a scalar (shape [1])
            return Self::from_elem(&[1], self.data.iter().next().unwrap().clone());
        }

        self.reshape(&new_shape)
            .expect("Squeeze should always succeed")
    }

    /// Remove a specific singleton dimension.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove (must have size 1)
    ///
    /// # Errors
    ///
    /// Returns an error if the axis doesn't have size 1.
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
    pub fn squeeze_axis(&self, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if self.shape()[axis] != 1 {
            anyhow::bail!(
                "Cannot squeeze axis {} with size {}",
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

    /// Add a singleton dimension at the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - Position where the new axis will be inserted
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
    /// let tensor = DenseND::<f64>::zeros(&[3, 5]);
    /// let unsqueezed = tensor.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[3, 1, 5]);
    /// ```
    pub fn unsqueeze(&self, axis: usize) -> anyhow::Result<Self> {
        if axis > self.rank() {
            anyhow::bail!(
                "Axis {} out of bounds for result rank {}",
                axis,
                self.rank() + 1
            );
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);

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
            // Convert scalar to 1D array
            self.reshape(&[1]).unwrap()
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
        match self.rank() {
            0 => self.reshape(&[1, 1]).unwrap(),
            1 => {
                let n = self.shape()[0];
                self.reshape(&[1, n]).unwrap()
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
        match self.rank() {
            0 => self.reshape(&[1, 1, 1]).unwrap(),
            1 => {
                let n = self.shape()[0];
                self.reshape(&[1, n, 1]).unwrap()
            }
            2 => {
                let m = self.shape()[0];
                let n = self.shape()[1];
                self.reshape(&[m, n, 1]).unwrap()
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
