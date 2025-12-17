//! Array manipulation operations (roll, flip, tile)
//!
//! This module provides methods for advanced array manipulation like
//! rolling elements, flipping along axes, and tiling/repeating arrays.

use super::types::DenseND;
use scirs2_core::numeric::{FromPrimitive, Num, NumCast};

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Roll array elements along a given axis
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let rolled = tensor.roll(1, 0).unwrap();
    ///
    /// assert_eq!(rolled[&[0]], 4.0);
    /// assert_eq!(rolled[&[1]], 1.0);
    /// assert_eq!(rolled[&[2]], 2.0);
    /// assert_eq!(rolled[&[3]], 3.0);
    /// ```
    pub fn roll(&self, shift: isize, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let axis_size = self.shape()[axis] as isize;
        let normalized_shift = ((shift % axis_size) + axis_size) % axis_size;

        if normalized_shift == 0 {
            return Ok(self.clone());
        }

        let mut result = Self::zeros(self.shape());
        let axis_len = self.shape()[axis];

        for i in 0..axis_len {
            let src_idx = i;
            let dst_idx = ((src_idx as isize + normalized_shift) % axis_size) as usize;

            // Copy slices along the axis
            let src_indices = super::functions::generate_indices(self.shape(), axis, src_idx);
            let dst_indices = super::functions::generate_indices(result.shape(), axis, dst_idx);

            for (src, dst) in src_indices.iter().zip(dst_indices.iter()) {
                result[dst.as_slice()] = self[src.as_slice()].clone();
            }
        }

        Ok(result)
    }

    /// Reverse the order of elements along an axis
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    /// let flipped = tensor.flip(0).unwrap();
    ///
    /// assert_eq!(flipped[&[0]], 4.0);
    /// assert_eq!(flipped[&[1]], 3.0);
    /// assert_eq!(flipped[&[2]], 2.0);
    /// assert_eq!(flipped[&[3]], 1.0);
    /// ```
    pub fn flip(&self, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        // Manually flip by copying elements in reverse order
        let mut result = Self::zeros(self.shape());
        let axis_len = self.shape()[axis];

        for i in 0..axis_len {
            let src_idx = i;
            let dst_idx = axis_len - 1 - i;

            let src_indices = super::functions::generate_indices(self.shape(), axis, src_idx);
            let dst_indices = super::functions::generate_indices(result.shape(), axis, dst_idx);

            for (src, dst) in src_indices.iter().zip(dst_indices.iter()) {
                result[dst.as_slice()] = self[src.as_slice()].clone();
            }
        }

        Ok(result)
    }

    /// Construct an array by repeating the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    /// let tiled = tensor.tile(&[3]).unwrap();
    ///
    /// assert_eq!(tiled.shape(), &[6]);
    /// assert_eq!(tiled[&[0]], 1.0);
    /// assert_eq!(tiled[&[2]], 1.0);
    /// assert_eq!(tiled[&[4]], 1.0);
    /// ```
    pub fn tile(&self, reps: &[usize]) -> anyhow::Result<Self> {
        if reps.is_empty() {
            anyhow::bail!("Repetition counts cannot be empty");
        }

        // Align ranks
        let self_rank = self.rank();
        let reps_rank = reps.len();
        let result_rank = self_rank.max(reps_rank);

        let mut result_shape = vec![1; result_rank];
        let mut tile_reps = vec![1; result_rank];

        // Fill in the actual shape and reps
        for i in 0..result_rank {
            if i < self_rank {
                let idx = self_rank - 1 - i;
                result_shape[result_rank - 1 - i] = self.shape()[idx];
            }
            if i < reps_rank {
                let idx = reps_rank - 1 - i;
                tile_reps[result_rank - 1 - i] = reps[idx];
                result_shape[result_rank - 1 - i] *= reps[idx];
            }
        }

        // Create result tensor
        let mut result = Self::zeros(&result_shape);

        // Tile the data
        let flat_self = if self_rank < result_rank {
            self.reshape(
                &result_shape
                    .iter()
                    .zip(&tile_reps)
                    .map(|(s, r)| s / r)
                    .collect::<Vec<_>>(),
            )?
        } else {
            self.clone()
        };

        // Simple tiling by copying blocks
        for idx in 0..result.len() {
            let mut multi_idx = vec![0; result_rank];
            let mut temp = idx;
            for i in (0..result_rank).rev() {
                multi_idx[i] = temp % result_shape[i];
                temp /= result_shape[i];
            }

            let src_idx: Vec<usize> = multi_idx
                .iter()
                .zip(tile_reps.iter())
                .map(|(idx, rep)| idx % (result_shape[multi_idx.len() - 1] / rep))
                .collect();

            result[multi_idx.as_slice()] = flat_self[src_idx.as_slice()].clone();
        }

        Ok(result)
    }

    /// Sort the elements along an axis.
    ///
    /// Returns a new tensor with elements sorted in ascending order along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to sort
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
    /// let tensor = DenseND::<f64>::from_vec(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]).unwrap();
    /// let sorted = tensor.sort(1).unwrap();
    ///
    /// // Row 0: [1, 2, 3]
    /// // Row 1: [4, 5, 6]
    /// assert_eq!(sorted[&[0, 0]], 1.0);
    /// assert_eq!(sorted[&[0, 1]], 2.0);
    /// assert_eq!(sorted[&[0, 2]], 3.0);
    /// ```
    pub fn sort(&self, axis: usize) -> anyhow::Result<Self>
    where
        T: PartialOrd,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let mut result = self.clone();
        let shape = self.shape().to_vec();
        let axis_size = shape[axis];

        // Sort along the axis
        let num_slices = self.len() / axis_size;

        for slice_idx in 0..num_slices {
            // Collect values along this slice
            let mut values = Vec::with_capacity(axis_size);

            for pos in 0..axis_size {
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = slice_idx;

                for (dim_idx, _) in shape.iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(pos);
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
                        indices.push(idx);
                    }
                }

                values.push(self.data[&indices[..]].clone());
            }

            // Sort the values
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Write sorted values back
            for (pos, value) in values.into_iter().enumerate() {
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = slice_idx;

                for (dim_idx, _) in shape.iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(pos);
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
                        indices.push(idx);
                    }
                }

                result.data[&indices[..]] = value;
            }
        }

        Ok(result)
    }

    /// Returns the indices that would sort the tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute sorting indices
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
    /// let tensor = DenseND::<f64>::from_vec(vec![3.0, 1.0, 2.0], &[3]).unwrap();
    /// let indices = tensor.argsort(0).unwrap();
    ///
    /// // Indices that sort [3, 1, 2] to [1, 2, 3] are [1, 2, 0]
    /// assert_eq!(indices[&[0]], 1);
    /// assert_eq!(indices[&[1]], 2);
    /// assert_eq!(indices[&[2]], 0);
    /// ```
    pub fn argsort(&self, axis: usize) -> anyhow::Result<DenseND<usize>>
    where
        T: PartialOrd,
    {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        let shape = self.shape().to_vec();
        let axis_size = shape[axis];
        let num_slices = self.len() / axis_size;

        let mut result = DenseND::<usize>::zeros(&shape);

        for slice_idx in 0..num_slices {
            // Collect values with their original indices
            let mut indexed_values: Vec<(usize, T)> = Vec::with_capacity(axis_size);

            for pos in 0..axis_size {
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = slice_idx;

                for (dim_idx, _) in shape.iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(pos);
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
                        indices.push(idx);
                    }
                }

                indexed_values.push((pos, self.data[&indices[..]].clone()));
            }

            // Sort by values, keeping track of original indices
            indexed_values.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

            // Extract the sorted indices
            let sorted_indices: Vec<usize> = indexed_values.iter().map(|(idx, _)| *idx).collect();

            // Write indices back
            for (pos, orig_idx) in sorted_indices.into_iter().enumerate() {
                let mut indices = Vec::with_capacity(self.rank());
                let mut remaining = slice_idx;

                for (dim_idx, _) in shape.iter().enumerate() {
                    if dim_idx == axis {
                        indices.push(pos);
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
                        indices.push(idx);
                    }
                }

                result.data[&indices[..]] = orig_idx;
            }
        }

        Ok(result)
    }

    /// Pad array with values according to the specified mode.
    ///
    /// # Arguments
    ///
    /// * `pad_width` - Number of values padded to the edges of each axis. Format: `[(before, after)]` for each axis.
    /// * `mode` - Padding mode (`PadMode::Constant`, `PadMode::Edge`, `PadMode::Reflect`, `PadMode::Wrap`)
    /// * `constant_value` - Value to use for constant padding (ignored for other modes)
    ///
    /// # Modes
    ///
    /// - `Constant`: Pads with a constant value
    /// - `Edge`: Pads with edge values of the array
    /// - `Reflect`: Pads with reflection of the array (values mirrored at edges)
    /// - `Wrap`: Pads with wrap-around (periodic boundary)
    ///
    /// # Complexity
    ///
    /// O(n) where n is the size of the padded array
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::{DenseND, PadMode};
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    ///
    /// // Constant padding
    /// let padded = tensor.pad(&[(1, 1)], PadMode::Constant, 0.0).unwrap();
    /// assert_eq!(padded.shape(), &[5]);
    /// assert_eq!(padded[&[0]], 0.0);
    /// assert_eq!(padded[&[1]], 1.0);
    /// assert_eq!(padded[&[4]], 0.0);
    ///
    /// // Edge padding
    /// let padded = tensor.pad(&[(1, 1)], PadMode::Edge, 0.0).unwrap();
    /// assert_eq!(padded[&[0]], 1.0);  // Edge value
    /// assert_eq!(padded[&[4]], 3.0);  // Edge value
    /// ```
    pub fn pad(
        &self,
        pad_width: &[(usize, usize)],
        mode: crate::PadMode,
        constant_value: T,
    ) -> anyhow::Result<Self> {
        if pad_width.len() != self.rank() {
            anyhow::bail!(
                "pad_width length {} does not match tensor rank {}",
                pad_width.len(),
                self.rank()
            );
        }

        // Calculate new shape
        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .zip(pad_width.iter())
            .map(|(&dim, &(before, after))| dim + before + after)
            .collect();

        // Create result tensor
        let mut result = match mode {
            crate::PadMode::Constant => Self::from_elem(&new_shape, constant_value),
            _ => Self::zeros(&new_shape),
        };

        // Copy original data to center
        self.copy_to_padded(&mut result, pad_width)?;

        // Fill padding based on mode
        match mode {
            crate::PadMode::Constant => {
                // Already filled with constant value
            }
            crate::PadMode::Edge => {
                self.pad_edge(&mut result, pad_width)?;
            }
            crate::PadMode::Reflect => {
                self.pad_reflect(&mut result, pad_width)?;
            }
            crate::PadMode::Wrap => {
                self.pad_wrap(&mut result, pad_width)?;
            }
        }

        Ok(result)
    }

    /// Helper: Copy original data to center of padded array
    fn copy_to_padded(
        &self,
        result: &mut Self,
        pad_width: &[(usize, usize)],
    ) -> anyhow::Result<()> {
        // Build offset for starting position
        let offsets: Vec<usize> = pad_width.iter().map(|(before, _)| *before).collect();

        // Recursively copy all elements
        fn copy_recursive<T>(
            src: &DenseND<T>,
            dst: &mut DenseND<T>,
            offsets: &[usize],
            src_idx: &mut Vec<usize>,
            current_dim: usize,
        ) where
            T: Clone + Num,
        {
            if current_dim == src.rank() {
                // Base case: copy the value
                let dst_idx: Vec<usize> = src_idx
                    .iter()
                    .zip(offsets.iter())
                    .map(|(&s, &o)| s + o)
                    .collect();
                dst[dst_idx.as_slice()] = src[src_idx.as_slice()].clone();
                return;
            }

            // Recursive case: iterate through current dimension
            for i in 0..src.shape()[current_dim] {
                src_idx[current_dim] = i;
                copy_recursive(src, dst, offsets, src_idx, current_dim + 1);
            }
        }

        let mut src_idx = vec![0; self.rank()];
        copy_recursive(self, result, &offsets, &mut src_idx, 0);

        Ok(())
    }

    /// Helper: Fill padding with edge values
    fn pad_edge(&self, result: &mut Self, pad_width: &[(usize, usize)]) -> anyhow::Result<()> {
        for axis in 0..self.rank() {
            let (before, after) = pad_width[axis];
            let axis_size = self.shape()[axis];

            // Pad before
            for pad_idx in 0..before {
                self.copy_slice(result, axis, 0, pad_idx, pad_width)?;
            }

            // Pad after
            for pad_idx in 0..after {
                let dst_idx = before + axis_size + pad_idx;
                self.copy_slice(result, axis, axis_size - 1, dst_idx, pad_width)?;
            }
        }

        Ok(())
    }

    /// Helper: Fill padding with reflected values
    fn pad_reflect(&self, result: &mut Self, pad_width: &[(usize, usize)]) -> anyhow::Result<()> {
        for axis in 0..self.rank() {
            let (before, after) = pad_width[axis];
            let axis_size = self.shape()[axis];

            // Pad before (reflect)
            for pad_idx in 0..before {
                let src_offset = before - pad_idx;
                let src_idx = src_offset.min(axis_size - 1);
                self.copy_slice(result, axis, src_idx, pad_idx, pad_width)?;
            }

            // Pad after (reflect)
            for pad_idx in 0..after {
                let src_offset = after - pad_idx - 1;
                let src_idx = (axis_size - 2).saturating_sub(src_offset);
                let dst_idx = before + axis_size + pad_idx;
                self.copy_slice(result, axis, src_idx, dst_idx, pad_width)?;
            }
        }

        Ok(())
    }

    /// Helper: Fill padding with wrap-around values
    fn pad_wrap(&self, result: &mut Self, pad_width: &[(usize, usize)]) -> anyhow::Result<()> {
        for axis in 0..self.rank() {
            let (before, after) = pad_width[axis];
            let axis_size = self.shape()[axis];

            // Pad before (wrap)
            for pad_idx in 0..before {
                let src_idx = (axis_size - before + pad_idx) % axis_size;
                self.copy_slice(result, axis, src_idx, pad_idx, pad_width)?;
            }

            // Pad after (wrap)
            for pad_idx in 0..after {
                let src_idx = pad_idx % axis_size;
                let dst_idx = before + axis_size + pad_idx;
                self.copy_slice(result, axis, src_idx, dst_idx, pad_width)?;
            }
        }

        Ok(())
    }

    /// Helper: Copy a slice from original tensor to a position in padded tensor
    fn copy_slice(
        &self,
        result: &mut Self,
        axis: usize,
        src_pos: usize,
        dst_pos: usize,
        pad_width: &[(usize, usize)],
    ) -> anyhow::Result<()> {
        let src_indices = super::functions::generate_indices(self.shape(), axis, src_pos);

        for src_idx in src_indices {
            // Convert to padded coordinates
            let dst_idx: Vec<usize> = src_idx
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    if i == axis {
                        dst_pos
                    } else {
                        val + pad_width[i].0
                    }
                })
                .collect();

            result[dst_idx.as_slice()] = self[src_idx.as_slice()].clone();
        }

        Ok(())
    }

    /// Find unique elements in the flattened tensor.
    ///
    /// Returns a 1D tensor containing sorted unique values.
    ///
    /// # Complexity
    ///
    /// O(n log n) due to sorting
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
    ///     &[6]
    /// ).unwrap();
    ///
    /// let unique = tensor.unique();
    /// assert_eq!(unique.shape(), &[3]);
    /// assert_eq!(unique[&[0]], 1.0);
    /// assert_eq!(unique[&[1]], 2.0);
    /// assert_eq!(unique[&[2]], 3.0);
    /// ```
    pub fn unique(&self) -> Self
    where
        T: PartialOrd,
    {
        let mut values: Vec<T> = self.as_slice().to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Remove duplicates
        values.dedup();

        let n = values.len();
        Self::from_vec(values, &[n]).unwrap_or_else(|_| {
            // Fallback to single element if dedup resulted in empty
            Self::from_elem(&[1], self.as_slice()[0].clone())
        })
    }

    /// Find unique elements and their counts.
    ///
    /// Returns a tuple of (unique_values, counts) where counts\[i\] is the number
    /// of occurrences of unique_values\[i\].
    ///
    /// # Complexity
    ///
    /// O(n log n) due to sorting
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
    ///     &[6]
    /// ).unwrap();
    ///
    /// let (unique, counts) = tensor.unique_with_counts();
    /// assert_eq!(unique.shape(), &[3]);
    /// assert_eq!(counts.shape(), &[3]);
    ///
    /// // Value 1.0 appears 3 times
    /// assert_eq!(unique[&[0]], 1.0);
    /// assert_eq!(counts[&[0]], 3);
    ///
    /// // Value 2.0 appears 2 times
    /// assert_eq!(unique[&[1]], 2.0);
    /// assert_eq!(counts[&[1]], 2);
    ///
    /// // Value 3.0 appears 1 time
    /// assert_eq!(unique[&[2]], 3.0);
    /// assert_eq!(counts[&[2]], 1);
    /// ```
    pub fn unique_with_counts(&self) -> (Self, DenseND<usize>)
    where
        T: PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = self.as_slice().to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Count occurrences
        let mut unique_vals = Vec::new();
        let mut counts = Vec::new();

        if !values.is_empty() {
            let mut current = values[0].clone();
            let mut count = 1usize;

            for val in values.iter().skip(1) {
                if val == &current {
                    count += 1;
                } else {
                    unique_vals.push(current);
                    counts.push(count);
                    current = val.clone();
                    count = 1;
                }
            }

            // Push the last group
            unique_vals.push(current);
            counts.push(count);
        }

        let n = unique_vals.len().max(1);
        let unique_tensor = Self::from_vec(unique_vals, &[n])
            .unwrap_or_else(|_| Self::from_elem(&[1], values[0].clone()));
        let counts_tensor =
            DenseND::from_vec(counts, &[n]).unwrap_or_else(|_| DenseND::from_elem(&[1], 1usize));

        (unique_tensor, counts_tensor)
    }

    /// Repeat elements of an array along an axis.
    ///
    /// Each element is repeated `repeats` times along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `repeats` - Number of repetitions for each element
    /// * `axis` - Axis along which to repeat
    ///
    /// # Complexity
    ///
    /// O(n * repeats) where n is the number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let repeated = tensor.repeat(2, 0).unwrap();
    ///
    /// assert_eq!(repeated.shape(), &[6]);
    /// assert_eq!(repeated[&[0]], 1.0);
    /// assert_eq!(repeated[&[1]], 1.0);
    /// assert_eq!(repeated[&[2]], 2.0);
    /// assert_eq!(repeated[&[3]], 2.0);
    /// assert_eq!(repeated[&[4]], 3.0);
    /// assert_eq!(repeated[&[5]], 3.0);
    /// ```
    pub fn repeat(&self, repeats: usize, axis: usize) -> anyhow::Result<Self> {
        if axis >= self.rank() {
            anyhow::bail!("Axis {} out of bounds for rank {}", axis, self.rank());
        }

        if repeats == 0 {
            anyhow::bail!("Repeats must be greater than 0");
        }

        if repeats == 1 {
            return Ok(self.clone());
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[axis] *= repeats;

        let mut result = Self::zeros(&new_shape);

        // Repeat each slice along the axis
        for i in 0..self.shape()[axis] {
            let src_indices = super::functions::generate_indices(self.shape(), axis, i);

            for r in 0..repeats {
                let dst_idx = i * repeats + r;
                let dst_indices = super::functions::generate_indices(&new_shape, axis, dst_idx);

                for (src, dst) in src_indices.iter().zip(dst_indices.iter()) {
                    result[dst.as_slice()] = self[src.as_slice()].clone();
                }
            }
        }

        Ok(result)
    }

    /// Repeat the entire array multiple times along a new axis.
    ///
    /// Unlike `repeat`, this repeats the entire array, not individual elements.
    ///
    /// # Arguments
    ///
    /// * `repeats` - Number of times to repeat the array
    /// * `axis` - Position where the new axis should be inserted
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    /// let tiled = tensor.repeat_array(3, 0).unwrap();
    ///
    /// assert_eq!(tiled.shape(), &[3, 2]);
    /// // Row 0: [1.0, 2.0]
    /// // Row 1: [1.0, 2.0]
    /// // Row 2: [1.0, 2.0]
    /// assert_eq!(tiled[&[0, 0]], 1.0);
    /// assert_eq!(tiled[&[1, 0]], 1.0);
    /// assert_eq!(tiled[&[2, 0]], 1.0);
    /// ```
    pub fn repeat_array(&self, repeats: usize, axis: usize) -> anyhow::Result<Self> {
        if axis > self.rank() {
            anyhow::bail!(
                "Axis {} out of bounds for inserting in rank {}",
                axis,
                self.rank()
            );
        }

        if repeats == 0 {
            anyhow::bail!("Repeats must be greater than 0");
        }

        // First unsqueeze to add dimension
        let with_axis = self.unsqueeze(axis)?;

        // Then repeat along that axis
        with_axis.repeat(repeats, axis)
    }
}

// Windowing operations requiring additional trait bounds
impl<T> DenseND<T>
where
    T: Clone + Num + NumCast + FromPrimitive,
{
    /// Create a sliding window view over the tensor along a specific axis
    ///
    /// This operation extracts overlapping windows of a specified size with a given stride.
    /// This is essential for convolution operations and sliding window analysis.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of the window
    /// * `stride` - Step size between consecutive windows
    /// * `axis` - The axis along which to create windows (default: 0)
    ///
    /// # Returns
    ///
    /// A new tensor with an additional dimension for windows. For a 1D input of shape \[n\],
    /// returns shape \[num_windows, window_size\]. For a 2D input of shape \[h, w\],
    /// returns shape \[num_windows_h, window_size, w\] when sliding along axis 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // 1D sliding window
    /// let data = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    /// let windows = data.sliding_window(3, 1, 0).unwrap();
    /// assert_eq!(windows.shape(), &[3, 3]);  // 3 windows of size 3
    /// // Window 0: [1.0, 2.0, 3.0]
    /// // Window 1: [2.0, 3.0, 4.0]
    /// // Window 2: [3.0, 4.0, 5.0]
    /// ```
    pub fn sliding_window(
        &self,
        window_size: usize,
        stride: usize,
        axis: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            axis < self.rank(),
            "Axis {} out of bounds for rank {}",
            axis,
            self.rank()
        );
        anyhow::ensure!(window_size > 0, "Window size must be > 0");
        anyhow::ensure!(stride > 0, "Stride must be > 0");

        let axis_size = self.shape()[axis];
        anyhow::ensure!(
            window_size <= axis_size,
            "Window size {} larger than axis size {}",
            window_size,
            axis_size
        );

        // Calculate number of windows
        let num_windows = (axis_size - window_size) / stride + 1;

        // Build output shape: [num_windows, window_size, ...other dims...]
        let mut output_shape = Vec::new();
        output_shape.push(num_windows);
        output_shape.push(window_size);
        for (i, &dim) in self.shape().iter().enumerate() {
            if i != axis {
                output_shape.push(dim);
            }
        }

        // Calculate total size
        let total_size: usize = output_shape.iter().product();
        let mut window_data = Vec::with_capacity(total_size);

        // Extract windows
        for window_idx in 0..num_windows {
            let start = window_idx * stride;
            for offset in 0..window_size {
                let pos = start + offset;

                // Extract slice at this position along the axis
                if self.rank() == 1 {
                    // Simple 1D case
                    window_data.push(self[&[pos]].clone());
                } else {
                    // Multi-dimensional case
                    // We need to iterate over all other dimensions
                    let mut indices = vec![0; self.rank()];
                    let mut sizes = self.shape().to_vec();
                    sizes[axis] = 1; // We're taking a single slice along this axis

                    fn iterate_indices<T>(
                        tensor: &DenseND<T>,
                        axis: usize,
                        pos: usize,
                        current_dim: usize,
                        indices: &mut [usize],
                        data: &mut Vec<T>,
                    ) where
                        T: Clone + Num + NumCast + FromPrimitive,
                    {
                        if current_dim == tensor.rank() {
                            // Base case: copy the value
                            data.push(tensor[indices].clone());
                            return;
                        }

                        if current_dim == axis {
                            // Skip the window axis and use pos
                            indices[current_dim] = pos;
                            iterate_indices(tensor, axis, pos, current_dim + 1, indices, data);
                        } else {
                            // Iterate through all values in this dimension
                            for i in 0..tensor.shape()[current_dim] {
                                indices[current_dim] = i;
                                iterate_indices(tensor, axis, pos, current_dim + 1, indices, data);
                            }
                        }
                    }

                    iterate_indices(self, axis, pos, 0, &mut indices, &mut window_data);
                }
            }
        }

        Self::from_vec(window_data, &output_shape)
    }

    /// Create a strided view with custom strides along each axis
    ///
    /// This operation downsamples the tensor by taking every nth element along each axis.
    /// This is useful for pooling operations and efficient downsampling.
    ///
    /// # Arguments
    ///
    /// * `strides` - Stride for each axis (must have same length as rank)
    ///
    /// # Returns
    ///
    /// A new tensor with reduced dimensions based on the strides
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     &[3, 3]
    /// ).unwrap();
    ///
    /// // Take every 2nd element along each axis
    /// let strided = data.strided_view(&[2, 2]).unwrap();
    /// assert_eq!(strided.shape(), &[2, 2]);
    /// // Result: [[1.0, 3.0], [7.0, 9.0]]
    /// ```
    pub fn strided_view(&self, strides: &[usize]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            strides.len() == self.rank(),
            "Strides length {} must match rank {}",
            strides.len(),
            self.rank()
        );

        for (i, &s) in strides.iter().enumerate() {
            anyhow::ensure!(s > 0, "Stride for axis {} must be > 0", i);
        }

        // Calculate output shape
        let mut output_shape = Vec::with_capacity(self.rank());
        for (i, &dim) in self.shape().iter().enumerate() {
            let new_dim = dim.div_ceil(strides[i]);
            output_shape.push(new_dim);
        }

        let total_size: usize = output_shape.iter().product();
        let mut strided_data = Vec::with_capacity(total_size);

        // Generate all output indices
        let mut output_idx = vec![0; self.rank()];

        fn generate_strided<T>(
            tensor: &DenseND<T>,
            strides: &[usize],
            current_dim: usize,
            indices: &mut [usize],
            output_shape: &[usize],
            data: &mut Vec<T>,
        ) where
            T: Clone + Num + NumCast + FromPrimitive,
        {
            if current_dim == tensor.rank() {
                // Compute input indices by multiplying by strides
                let input_idx: Vec<usize> = indices
                    .iter()
                    .zip(strides.iter())
                    .map(|(&i, &s)| i * s)
                    .collect();
                data.push(tensor[&input_idx].clone());
                return;
            }

            for i in 0..output_shape[current_dim] {
                indices[current_dim] = i;
                generate_strided(
                    tensor,
                    strides,
                    current_dim + 1,
                    indices,
                    output_shape,
                    data,
                );
            }
        }

        generate_strided(
            self,
            strides,
            0,
            &mut output_idx,
            &output_shape,
            &mut strided_data,
        );

        Self::from_vec(strided_data, &output_shape)
    }

    /// Extract 2D patches from a 2D tensor (image) with given patch size and stride
    ///
    /// This is a specialized version of sliding_window for 2D data (images),
    /// which is commonly used in computer vision and convolutional neural networks.
    ///
    /// # Arguments
    ///
    /// * `patch_size` - (height, width) of each patch
    /// * `stride` - (stride_h, stride_w) step size
    ///
    /// # Returns
    ///
    /// A 4D tensor of shape [num_patches_h, num_patches_w, patch_h, patch_w]
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // Create a 4x4 image
    /// let img = DenseND::<f64>::from_vec(
    ///     (1..=16).map(|x| x as f64).collect(),
    ///     &[4, 4]
    /// ).unwrap();
    ///
    /// // Extract 2x2 patches with stride 2 (non-overlapping)
    /// let patches = img.extract_patches((2, 2), (2, 2)).unwrap();
    /// assert_eq!(patches.shape(), &[2, 2, 2, 2]);  // 2x2 grid of 2x2 patches
    /// ```
    pub fn extract_patches(
        &self,
        patch_size: (usize, usize),
        stride: (usize, usize),
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.rank() == 2,
            "extract_patches requires 2D tensor, got rank {}",
            self.rank()
        );

        let (patch_h, patch_w) = patch_size;
        let (stride_h, stride_w) = stride;
        let (img_h, img_w) = (self.shape()[0], self.shape()[1]);

        anyhow::ensure!(patch_h > 0 && patch_w > 0, "Patch size must be > 0");
        anyhow::ensure!(stride_h > 0 && stride_w > 0, "Stride must be > 0");
        anyhow::ensure!(
            patch_h <= img_h && patch_w <= img_w,
            "Patch size ({}, {}) larger than image size ({}, {})",
            patch_h,
            patch_w,
            img_h,
            img_w
        );

        // Calculate number of patches in each dimension
        let num_patches_h = (img_h - patch_h) / stride_h + 1;
        let num_patches_w = (img_w - patch_w) / stride_w + 1;

        let output_shape = [num_patches_h, num_patches_w, patch_h, patch_w];
        let total_size = num_patches_h * num_patches_w * patch_h * patch_w;
        let mut patch_data = Vec::with_capacity(total_size);

        // Extract each patch
        for patch_i in 0..num_patches_h {
            for patch_j in 0..num_patches_w {
                let start_i = patch_i * stride_h;
                let start_j = patch_j * stride_w;

                // Copy patch
                for i in 0..patch_h {
                    for j in 0..patch_w {
                        let val = self[&[start_i + i, start_j + j]].clone();
                        patch_data.push(val);
                    }
                }
            }
        }

        Self::from_vec(patch_data, &output_shape)
    }

    /// Perform 2D max pooling on a 2D tensor
    ///
    /// Applies maximum pooling with specified kernel size and stride.
    /// This is a fundamental operation in convolutional neural networks.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - (height, width) of the pooling kernel
    /// * `stride` - (stride_h, stride_w) step size (defaults to kernel_size if None)
    ///
    /// # Returns
    ///
    /// A tensor with reduced spatial dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // Create a 4x4 input
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0,
    ///          5.0, 6.0, 7.0, 8.0,
    ///          9.0, 10.0, 11.0, 12.0,
    ///          13.0, 14.0, 15.0, 16.0],
    ///     &[4, 4]
    /// ).unwrap();
    ///
    /// // 2x2 max pooling with stride 2
    /// let pooled = data.max_pool_2d((2, 2), None).unwrap();
    /// assert_eq!(pooled.shape(), &[2, 2]);
    /// assert_eq!(pooled[&[0, 0]], 6.0);  // max of [1,2,5,6]
    /// assert_eq!(pooled[&[0, 1]], 8.0);  // max of [3,4,7,8]
    /// ```
    pub fn max_pool_2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> anyhow::Result<Self>
    where
        T: PartialOrd,
    {
        anyhow::ensure!(
            self.rank() == 2,
            "max_pool_2d requires 2D tensor, got rank {}",
            self.rank()
        );

        let (kernel_h, kernel_w) = kernel_size;
        let (stride_h, stride_w) = stride.unwrap_or(kernel_size);
        let (input_h, input_w) = (self.shape()[0], self.shape()[1]);

        anyhow::ensure!(kernel_h > 0 && kernel_w > 0, "Kernel size must be > 0");
        anyhow::ensure!(stride_h > 0 && stride_w > 0, "Stride must be > 0");
        anyhow::ensure!(
            kernel_h <= input_h && kernel_w <= input_w,
            "Kernel size ({}, {}) larger than input size ({}, {})",
            kernel_h,
            kernel_w,
            input_h,
            input_w
        );

        // Calculate output dimensions
        let output_h = (input_h - kernel_h) / stride_h + 1;
        let output_w = (input_w - kernel_w) / stride_w + 1;

        let mut pooled_data = Vec::with_capacity(output_h * output_w);

        // Perform max pooling
        for out_i in 0..output_h {
            for out_j in 0..output_w {
                let start_i = out_i * stride_h;
                let start_j = out_j * stride_w;

                // Find maximum in the kernel window
                let mut max_val = self[&[start_i, start_j]].clone();
                for i in 0..kernel_h {
                    for j in 0..kernel_w {
                        let val = &self[&[start_i + i, start_j + j]];
                        if val > &max_val {
                            max_val = val.clone();
                        }
                    }
                }
                pooled_data.push(max_val);
            }
        }

        Self::from_vec(pooled_data, &[output_h, output_w])
    }

    /// Perform 2D average pooling on a 2D tensor
    ///
    /// Applies average pooling with specified kernel size and stride.
    /// This is commonly used in convolutional neural networks for downsampling.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - (height, width) of the pooling kernel
    /// * `stride` - (stride_h, stride_w) step size (defaults to kernel_size if None)
    ///
    /// # Returns
    ///
    /// A tensor with reduced spatial dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// // Create a 4x4 input
    /// let data = DenseND::<f64>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0,
    ///          5.0, 6.0, 7.0, 8.0,
    ///          9.0, 10.0, 11.0, 12.0,
    ///          13.0, 14.0, 15.0, 16.0],
    ///     &[4, 4]
    /// ).unwrap();
    ///
    /// // 2x2 average pooling with stride 2
    /// let pooled = data.avg_pool_2d((2, 2), None).unwrap();
    /// assert_eq!(pooled.shape(), &[2, 2]);
    /// assert_eq!(pooled[&[0, 0]], 3.5);  // avg of [1,2,5,6] = 14/4 = 3.5
    /// ```
    pub fn avg_pool_2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        anyhow::ensure!(
            self.rank() == 2,
            "avg_pool_2d requires 2D tensor, got rank {}",
            self.rank()
        );

        let (kernel_h, kernel_w) = kernel_size;
        let (stride_h, stride_w) = stride.unwrap_or(kernel_size);
        let (input_h, input_w) = (self.shape()[0], self.shape()[1]);

        anyhow::ensure!(kernel_h > 0 && kernel_w > 0, "Kernel size must be > 0");
        anyhow::ensure!(stride_h > 0 && stride_w > 0, "Stride must be > 0");
        anyhow::ensure!(
            kernel_h <= input_h && kernel_w <= input_w,
            "Kernel size ({}, {}) larger than input size ({}, {})",
            kernel_h,
            kernel_w,
            input_h,
            input_w
        );

        // Calculate output dimensions
        let output_h = (input_h - kernel_h) / stride_h + 1;
        let output_w = (input_w - kernel_w) / stride_w + 1;

        let mut pooled_data = Vec::with_capacity(output_h * output_w);
        let kernel_area = T::from_usize(kernel_h * kernel_w).unwrap();

        // Perform average pooling
        for out_i in 0..output_h {
            for out_j in 0..output_w {
                let start_i = out_i * stride_h;
                let start_j = out_j * stride_w;

                // Compute average in the kernel window
                let mut sum = T::zero();
                for i in 0..kernel_h {
                    for j in 0..kernel_w {
                        sum = sum + self[&[start_i + i, start_j + j]];
                    }
                }
                pooled_data.push(sum / kernel_area);
            }
        }

        Self::from_vec(pooled_data, &[output_h, output_w])
    }

    /// Perform 2D adaptive average pooling to a target output size
    ///
    /// Unlike regular pooling, adaptive pooling automatically determines the
    /// kernel size and stride to produce the desired output size.
    ///
    /// # Arguments
    ///
    /// * `output_size` - Desired (height, width) of the output
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let data = DenseND::<f64>::from_vec((1..=16).map(|x| x as f64).collect(), &[4, 4]).unwrap();
    ///
    /// // Adaptively pool to 2x2 output
    /// let pooled = data.adaptive_avg_pool_2d((2, 2)).unwrap();
    /// assert_eq!(pooled.shape(), &[2, 2]);
    /// ```
    pub fn adaptive_avg_pool_2d(&self, output_size: (usize, usize)) -> anyhow::Result<Self>
    where
        T: scirs2_core::numeric::Float,
    {
        anyhow::ensure!(
            self.rank() == 2,
            "adaptive_avg_pool_2d requires 2D tensor, got rank {}",
            self.rank()
        );

        let (input_h, input_w) = (self.shape()[0], self.shape()[1]);
        let (output_h, output_w) = output_size;

        anyhow::ensure!(output_h > 0 && output_w > 0, "Output size must be > 0");
        anyhow::ensure!(
            output_h <= input_h && output_w <= input_w,
            "Output size ({}, {}) must be <= input size ({}, {})",
            output_h,
            output_w,
            input_h,
            input_w
        );

        let mut pooled_data = Vec::with_capacity(output_h * output_w);

        for out_i in 0..output_h {
            for out_j in 0..output_w {
                // Calculate adaptive window for this output position
                let start_i = (out_i * input_h) / output_h;
                let end_i = ((out_i + 1) * input_h) / output_h;
                let start_j = (out_j * input_w) / output_w;
                let end_j = ((out_j + 1) * input_w) / output_w;

                // Compute average in the adaptive window
                let mut sum = T::zero();
                let mut count = 0;
                for i in start_i..end_i {
                    for j in start_j..end_j {
                        sum = sum + self[&[i, j]];
                        count += 1;
                    }
                }
                pooled_data.push(sum / T::from_usize(count).unwrap());
            }
        }

        Self::from_vec(pooled_data, &[output_h, output_w])
    }

    /// Resize a 2D tensor using nearest neighbor interpolation.
    ///
    /// # Arguments
    ///
    /// * `new_height` - Target height
    /// * `new_width` - Target width
    ///
    /// # Returns
    ///
    /// Resized 2D tensor with shape `[new_height, new_width]`
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D.
    ///
    /// # Complexity
    ///
    /// O(new_height × new_width)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let image = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let resized = image.resize_nearest(4, 4).unwrap();
    /// assert_eq!(resized.shape(), &[4, 4]);
    /// ```
    pub fn resize_nearest(&self, new_height: usize, new_width: usize) -> anyhow::Result<Self>
    where
        T: NumCast + FromPrimitive,
    {
        if self.rank() != 2 {
            anyhow::bail!(
                "resize_nearest requires a 2D tensor, got rank {}",
                self.rank()
            );
        }

        let [old_height, old_width] = [self.shape()[0], self.shape()[1]];
        let mut result_data = Vec::with_capacity(new_height * new_width);

        let height_scale = old_height as f64 / new_height as f64;
        let width_scale = old_width as f64 / new_width as f64;

        for i in 0..new_height {
            for j in 0..new_width {
                let src_i = ((i as f64 + 0.5) * height_scale).floor() as usize;
                let src_j = ((j as f64 + 0.5) * width_scale).floor() as usize;

                let src_i = src_i.min(old_height - 1);
                let src_j = src_j.min(old_width - 1);

                result_data.push(self[&[src_i, src_j]].clone());
            }
        }

        Self::from_vec(result_data, &[new_height, new_width])
    }

    /// Resize a 2D tensor using bilinear interpolation.
    ///
    /// # Arguments
    ///
    /// * `new_height` - Target height
    /// * `new_width` - Target width
    ///
    /// # Returns
    ///
    /// Resized 2D tensor with shape `[new_height, new_width]`
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D.
    ///
    /// # Complexity
    ///
    /// O(new_height × new_width)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::DenseND;
    ///
    /// let image = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let resized = image.resize_bilinear(4, 4).unwrap();
    /// assert_eq!(resized.shape(), &[4, 4]);
    /// ```
    pub fn resize_bilinear(&self, new_height: usize, new_width: usize) -> anyhow::Result<Self>
    where
        T: NumCast + FromPrimitive + scirs2_core::numeric::Float,
    {
        if self.rank() != 2 {
            anyhow::bail!(
                "resize_bilinear requires a 2D tensor, got rank {}",
                self.rank()
            );
        }

        let [old_height, old_width] = [self.shape()[0], self.shape()[1]];
        let mut result_data = Vec::with_capacity(new_height * new_width);

        let height_scale = (old_height - 1) as f64 / (new_height - 1).max(1) as f64;
        let width_scale = (old_width - 1) as f64 / (new_width - 1).max(1) as f64;

        for i in 0..new_height {
            for j in 0..new_width {
                let src_i = i as f64 * height_scale;
                let src_j = j as f64 * width_scale;

                let i0 = src_i.floor() as usize;
                let i1 = (i0 + 1).min(old_height - 1);
                let j0 = src_j.floor() as usize;
                let j1 = (j0 + 1).min(old_width - 1);

                let di = T::from_f64(src_i - i0 as f64).unwrap();
                let dj = T::from_f64(src_j - j0 as f64).unwrap();
                let one = T::one();

                // Bilinear interpolation
                let v00 = self[&[i0, j0]];
                let v01 = self[&[i0, j1]];
                let v10 = self[&[i1, j0]];
                let v11 = self[&[i1, j1]];

                let value = v00 * (one - di) * (one - dj)
                    + v01 * (one - di) * dj
                    + v10 * di * (one - dj)
                    + v11 * di * dj;

                result_data.push(value);
            }
        }

        Self::from_vec(result_data, &[new_height, new_width])
    }
}
