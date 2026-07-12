//! Shape manipulation operations on tensors
//!
//! This module provides comprehensive shape manipulation including reshape, permute,
//! unfold/fold (matricization/tensorization), squeeze/unsqueeze, and axis operations.

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array2, ArrayView, IxDyn};
use scirs2_core::numeric::Num;
use smallvec::{smallvec, SmallVec};

impl<T> DenseND<T>
where
    T: Clone + Num,
{
    /// Validate that `axes` is a permutation of `0..rank`.
    ///
    /// Shared by [`DenseND::permute`], [`DenseND::permute_view`] and
    /// [`DenseND::into_permuted`] so that all three reject exactly the same
    /// inputs with exactly the same messages.
    ///
    /// The `seen` scratch buffer is a `SmallVec` with inline capacity 8: for the
    /// tensor ranks that occur in practice this is stack-only, which is what
    /// lets the zero-copy variants be *genuinely* allocation-free.
    ///
    /// # Complexity
    ///
    /// O(rank), no heap allocation for rank <= 8.
    fn validate_permutation(&self, axes: &[usize]) -> anyhow::Result<()> {
        let rank = self.rank();
        if axes.len() != rank {
            anyhow::bail!(
                "Permutation axes length {} does not match tensor rank {}",
                axes.len(),
                rank
            );
        }
        let mut seen: SmallVec<[bool; 8]> = smallvec![false; rank];
        for &axis in axes {
            if axis >= rank {
                anyhow::bail!("Invalid axis {} for rank {}", axis, rank);
            }
            if seen[axis] {
                anyhow::bail!("Duplicate axis {} in permutation", axis);
            }
            seen[axis] = true;
        }
        Ok(())
    }

    /// Reshape the tensor to a new shape, returning an independent owned tensor.
    ///
    /// # Copy behaviour
    ///
    /// This borrowing variant **always copies** the elements exactly once — it
    /// has to, because the returned tensor owns its buffer and must stay valid
    /// (and independent) after `self` is mutated or dropped. Concretely it costs
    /// one allocation of `len() * size_of::<T>()` bytes on every call, whether
    /// or not `self` is contiguous:
    ///
    /// * contiguous `self`: one linear `memcpy`;
    /// * non-contiguous `self` (e.g. the result of [`DenseND::permute`]): the
    ///   elements are gathered in row-major *logical* order, matching NumPy's
    ///   `reshape` semantics.
    ///
    /// If you do not need `self` afterwards, prefer [`DenseND::into_reshape`],
    /// which consumes the tensor and is genuinely zero-copy (**0 allocations**)
    /// whenever the buffer is already contiguous.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The target shape
    ///
    /// # Returns
    ///
    /// A reshaped tensor, or an error if the total size doesn't match
    ///
    /// # Errors
    ///
    /// Returns an error if the element count of `new_shape` differs from
    /// `self.len()`.
    ///
    /// # Complexity
    ///
    /// O(n) time, O(n) extra space.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::zeros(&[2, 3, 4]);
    /// let reshaped = tensor.reshape(&[6, 4]).unwrap();
    /// assert_eq!(reshaped.shape(), &[6, 4]);
    /// // `tensor` is still usable: `reshaped` owns a separate buffer.
    /// assert_eq!(tensor.shape(), &[2, 3, 4]);
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

    /// Reshape the tensor by value, reusing its buffer.
    ///
    /// This is the zero-copy counterpart of [`DenseND::reshape`]. Because it
    /// consumes `self`, a contiguous tensor can simply have its shape/stride
    /// metadata rewritten and its buffer moved into the result: **0 allocations,
    /// no data movement, O(1)**.
    ///
    /// A non-contiguous tensor (e.g. one produced by [`DenseND::into_permuted`])
    /// still has to gather its elements in row-major logical order, costing one
    /// O(n) copy — exactly what [`DenseND::reshape`] would have done.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The target shape
    ///
    /// # Errors
    ///
    /// Returns an error if the element count of `new_shape` differs from
    /// `self.len()`.
    ///
    /// # Complexity
    ///
    /// O(1) for a contiguous tensor, O(n) otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec((0..24).map(f64::from).collect(), &[2, 3, 4]).unwrap();
    /// // Zero-copy: the buffer is moved, not duplicated.
    /// let reshaped = tensor.into_reshape(&[6, 4]).unwrap();
    /// assert_eq!(reshaped.shape(), &[6, 4]);
    /// assert_eq!(reshaped[&[0, 0]], 0.0);
    /// assert_eq!(reshaped[&[5, 3]], 23.0);
    /// ```
    pub fn into_reshape(self, new_shape: &[usize]) -> anyhow::Result<Self> {
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
        if self.data.is_standard_layout() {
            // Contiguous: `into_shape_with_order` rewrites shape/strides and
            // moves the existing buffer. Nothing is copied or allocated.
            let data = self
                .data
                .into_shape_with_order(IxDyn(new_shape))
                .map_err(|e| anyhow::anyhow!("Failed to reshape contiguous tensor: {}", e))?;
            Ok(Self { data })
        } else {
            // Non-contiguous: the logical row-major order does not match memory
            // order, so the elements must be gathered. One copy, same result as
            // `reshape`.
            let flat: Vec<T> = self.data.iter().cloned().collect();
            Self::from_vec(flat, new_shape)
        }
    }

    /// Permute (transpose) the axes of the tensor, returning an owned tensor.
    ///
    /// # Copy behaviour
    ///
    /// The returned tensor is an independent owner of its data, so this costs
    /// exactly one O(n) buffer copy; the axis permutation itself is pure
    /// shape/stride metadata and moves no data. The result is **non-contiguous**
    /// for any non-identity permutation (its strides are permuted), which means
    /// a subsequent [`DenseND::reshape`] on it must gather elements rather than
    /// take its metadata-only path.
    ///
    /// Two allocation-free alternatives exist when the copy is not wanted:
    ///
    /// * [`DenseND::permute_view`] — borrow `self` and get a zero-copy
    ///   `ArrayView` with permuted axes (O(1)).
    /// * [`DenseND::into_permuted`] — consume `self` and move its buffer into
    ///   the permuted tensor (O(1), 0 allocations).
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
    /// # Complexity
    ///
    /// O(n) time (the buffer copy), O(n) extra space.
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
        self.validate_permutation(axes)?;
        let permuted = self.data.clone().permuted_axes(IxDyn(axes));
        Ok(Self { data: permuted })
    }

    /// Borrow the tensor as a zero-copy view with permuted axes.
    ///
    /// `permuted_axes` only rewrites shape/stride metadata, so this is O(1) and
    /// allocation-free — no element is touched. Use it whenever the permuted
    /// tensor is only *read* (which is the case for every internal consumer,
    /// e.g. [`DenseND::unfold`]); use [`DenseND::permute`] only when an owned,
    /// independent tensor is genuinely required.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of axes (must be a permutation of 0..rank)
    ///
    /// # Errors
    ///
    /// Returns an error if `axes` is not a valid permutation.
    ///
    /// # Complexity
    ///
    /// O(1) time, O(1) space, **0 allocations**.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec((0..24).map(f64::from).collect(), &[2, 3, 4]).unwrap();
    /// let view = tensor.permute_view(&[2, 0, 1]).unwrap();
    /// assert_eq!(view.shape(), &[4, 2, 3]);
    /// // Same elements, no copy: view[k, i, j] == tensor[i, j, k]
    /// assert_eq!(view[[3, 1, 2]], tensor[&[1, 2, 3]]);
    /// ```
    pub fn permute_view(&self, axes: &[usize]) -> anyhow::Result<ArrayView<'_, T, IxDyn>> {
        self.validate_permutation(axes)?;
        Ok(self.data.view().permuted_axes(IxDyn(axes)))
    }

    /// Permute the axes by value, reusing the tensor's buffer.
    ///
    /// The zero-copy, owning counterpart of [`DenseND::permute`]: `self` is
    /// consumed, its buffer is moved into the result and only the shape/stride
    /// metadata is rewritten. O(1), **0 allocations**.
    ///
    /// As with [`DenseND::permute`], the result is non-contiguous for any
    /// non-identity permutation.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of axes (must be a permutation of 0..rank)
    ///
    /// # Errors
    ///
    /// Returns an error if `axes` is not a valid permutation.
    ///
    /// # Complexity
    ///
    /// O(1) time, O(1) space, **0 allocations**.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let tensor = DenseND::<f64>::from_vec((0..24).map(f64::from).collect(), &[2, 3, 4]).unwrap();
    /// let expected = tensor[&[1, 2, 3]];
    /// let permuted = tensor.into_permuted(&[2, 0, 1]).unwrap();
    /// assert_eq!(permuted.shape(), &[4, 2, 3]);
    /// assert_eq!(permuted[&[3, 1, 2]], expected);
    /// ```
    pub fn into_permuted(self, axes: &[usize]) -> anyhow::Result<Self> {
        self.validate_permutation(axes)?;
        let permuted = self.data.permuted_axes(IxDyn(axes));
        Ok(Self { data: permuted })
    }

    /// Unfold (matricize) the tensor along a specific mode.
    ///
    /// Mode-n unfolding arranges the mode-n fibers as columns of a matrix.
    /// This is critical for tensor decompositions (CP, Tucker, TT) — it runs on
    /// every mode of every CP-ALS / Tucker-HOOI iteration, so it is a hot path.
    ///
    /// # Copy behaviour
    ///
    /// Exactly **one** O(n) pass over the data. The mode-to-front permutation is
    /// taken as a zero-copy view ([`DenseND::permute_view`], pure metadata) and
    /// the elements are gathered straight into the output matrix's buffer.
    ///
    /// (The previous implementation went `permute` -> `reshape`, which cost
    /// *two* full copies: `permute` cloned the buffer, and then `reshape`'s
    /// metadata-only path could not fire — a permuted buffer is non-contiguous —
    /// so it fell back to a second gather.)
    ///
    /// When the permuted view keeps a unit-stride innermost axis (always true
    /// for a contiguous tensor with `mode != rank - 1`) each row of the gather is
    /// a contiguous slice, so the copy is a sequence of `memcpy`s rather than an
    /// element-at-a-time strided walk.
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
    /// # Complexity
    ///
    /// O(n) time, one allocation of `n * size_of::<T>()` bytes.
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
        let rank = self.rank();
        if mode >= rank {
            anyhow::bail!("Mode {} out of bounds for rank {}", mode, rank);
        }

        let shape = self.shape();
        let rows = shape[mode];
        let cols: usize = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != mode)
            .map(|(_, &s)| s)
            .product();

        // Permute so that `mode` becomes the leading axis. O(1): only the
        // shape/stride metadata of the *view* is rewritten, no data is moved.
        let mut perm: Vec<usize> = Vec::with_capacity(rank);
        perm.push(mode);
        perm.extend((0..mode).chain((mode + 1)..rank));
        let permuted = self.data.view().permuted_axes(IxDyn(&perm));

        let total = rows * cols;
        let mut flat: Vec<T> = Vec::with_capacity(total);

        // Fast path: the innermost axis of the permuted view still has unit
        // stride, so every innermost lane is a contiguous slice and the gather
        // becomes one `memcpy` per lane. `rows()` walks the lanes in row-major
        // logical order, which is precisely the order `iter()` would produce.
        if permuted.strides().last() == Some(&1) {
            for lane in permuted.rows() {
                match lane.as_slice() {
                    Some(contiguous) => flat.extend_from_slice(contiguous),
                    // A lane that is not a slice means the fast path does not
                    // apply after all; fall through to the general gather below.
                    None => break,
                }
            }
        }

        // General path (also the safety net if the fast path bailed out
        // part-way): gather in row-major logical order. Byte-identical output.
        if flat.len() != total {
            flat.clear();
            flat.extend(permuted.iter().cloned());
        }

        Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| anyhow::anyhow!("Failed to build mode-{} unfolding: {}", mode, e))
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

        // Reshape matrix to intermediate tensor (one copy out of the matrix).
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

        // `intermediate` is a freshly-built temporary that nobody else can see,
        // so move its buffer into the permuted result instead of cloning it.
        // Identical output (same buffer contents, same permuted strides) for one
        // fewer O(n) copy.
        intermediate.into_permuted(&inverse_perm)
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
    /// Returns a new owned 1-D tensor holding the elements in row-major (C)
    /// order. Costs one O(n) copy (see [`DenseND::reshape`]); this also means it
    /// works for non-contiguous tensors, such as the result of
    /// [`DenseND::permute`], where the elements have to be gathered.
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
    ///
    /// // Also correct for a non-contiguous (permuted) tensor: row-major order
    /// // of the *logical* element layout.
    /// let permuted = tensor.permute(&[1, 0]).unwrap();
    /// assert_eq!(permuted.flatten().to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn flatten(&self) -> Self {
        let total = self.len();
        // `reshape` handles both layouts: metadata-only view + copy when
        // contiguous, logical-order gather when not. A 1-D shape always has the
        // same element count as the source, so this cannot fail.
        self.reshape(&[total])
            .expect("flatten: [len] preserves the element count")
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
    use scirs2_core::ndarray_ext::{array, Ix2};

    /// Faithful reproduction of the *pre-optimization* `unfold`: permute (which
    /// clones the whole buffer) and then reshape (which, on the now
    /// non-contiguous buffer, gathers the elements a second time).
    ///
    /// It is written purely against the public API, so it is an independent
    /// oracle: the fused single-pass `unfold` must agree with it bit-for-bit for
    /// every shape/mode, contiguous input or not.
    fn legacy_unfold(tensor: &DenseND<f64>, mode: usize) -> Array2<f64> {
        let shape = tensor.shape().to_vec();
        let rows = shape[mode];
        let cols: usize = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != mode)
            .map(|(_, &s)| s)
            .product();

        let mut perm: Vec<usize> = vec![mode];
        perm.extend((0..mode).chain((mode + 1)..shape.len()));

        let permuted = tensor.permute(&perm).expect("valid permutation");
        let reshaped = permuted.reshape(&[rows, cols]).expect("size preserved");
        reshaped
            .as_array()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("2-D by construction")
    }

    fn iota(shape: &[usize]) -> DenseND<f64> {
        let total: usize = shape.iter().product();
        let data: Vec<f64> = (0..total).map(|i| i as f64).collect();
        DenseND::from_vec(data, shape).expect("valid shape")
    }

    // -- unfold: byte-identical to the two-copy implementation ----------

    #[test]
    fn test_unfold_matches_legacy_implementation_all_shapes_and_modes() {
        let shapes: &[&[usize]] = &[
            &[7],
            &[2, 3],
            &[3, 2],
            &[2, 3, 4],
            &[4, 3, 2],
            &[5, 1, 4],
            &[1, 5, 1],
            &[4, 3, 2, 5],
            &[2, 2, 2, 2, 3],
        ];

        for shape in shapes {
            let tensor = iota(shape);
            for mode in 0..shape.len() {
                let fused = tensor.unfold(mode).expect("mode in range");
                let legacy = legacy_unfold(&tensor, mode);
                assert_eq!(
                    fused.shape(),
                    legacy.shape(),
                    "shape mismatch for shape {shape:?} mode {mode}"
                );
                assert_eq!(
                    fused, legacy,
                    "unfold output changed for shape {shape:?} mode {mode}"
                );
            }
        }
    }

    #[test]
    fn test_unfold_matches_legacy_on_non_contiguous_input() {
        // A permuted tensor is non-contiguous, which exercises the general
        // gather path (its innermost stride is not 1 for every permutation).
        let base = iota(&[4, 3, 2]);
        for perm in [[2, 0, 1], [1, 2, 0], [2, 1, 0], [0, 2, 1]] {
            let permuted = base.permute(&perm).expect("valid permutation");
            assert!(
                !permuted.is_contiguous(),
                "test premise: permuting must yield a non-contiguous tensor"
            );
            for mode in 0..3 {
                let fused = permuted.unfold(mode).expect("mode in range");
                let legacy = legacy_unfold(&permuted, mode);
                assert_eq!(fused, legacy, "perm {perm:?} mode {mode}");
            }
        }
    }

    #[test]
    fn test_unfold_known_values() {
        // Textbook mode-1 unfolding of a 2x2x2 tensor (Kolda & Bader
        // conventions with row-major fibers).
        let tensor = iota(&[2, 2, 2]);
        let unfolded = tensor.unfold(1).expect("mode 1");
        assert_eq!(unfolded.shape(), &[2, 4]);
        // Row j collects every element whose mode-1 index is j, in row-major
        // order of the remaining (mode-0, mode-2) axes.
        assert_eq!(unfolded[[0, 0]], tensor[&[0, 0, 0]]);
        assert_eq!(unfolded[[0, 1]], tensor[&[0, 0, 1]]);
        assert_eq!(unfolded[[0, 2]], tensor[&[1, 0, 0]]);
        assert_eq!(unfolded[[0, 3]], tensor[&[1, 0, 1]]);
        assert_eq!(unfolded[[1, 0]], tensor[&[0, 1, 0]]);
        assert_eq!(unfolded[[1, 3]], tensor[&[1, 1, 1]]);
    }

    #[test]
    fn test_unfold_last_mode_uses_strided_path() {
        // mode == rank - 1 leaves the permuted view with a non-unit innermost
        // stride, so the `memcpy`-per-lane fast path must bail out cleanly and
        // the general gather must still produce the right answer.
        let tensor = iota(&[3, 4, 5]);
        let fused = tensor.unfold(2).expect("mode 2");
        let legacy = legacy_unfold(&tensor, 2);
        assert_eq!(fused, legacy);
        assert_eq!(fused.shape(), &[5, 12]);
    }

    #[test]
    fn test_unfold_fold_roundtrip_all_modes() {
        let shape = [4, 3, 2];
        let tensor = iota(&shape);
        for mode in 0..3 {
            let unfolded = tensor.unfold(mode).expect("mode in range");
            let folded = DenseND::fold(&unfolded, &shape, mode).expect("compatible");
            assert_eq!(folded.shape(), tensor.shape());
            assert_eq!(
                folded.to_vec(),
                tensor.to_vec(),
                "roundtrip failed at mode {mode}"
            );
        }
    }

    #[test]
    fn test_unfold_mode_out_of_bounds() {
        let tensor = iota(&[2, 3]);
        assert!(tensor.unfold(2).is_err());
    }

    // -- into_reshape ---------------------------------------------------

    #[test]
    fn test_into_reshape_matches_reshape_contiguous() {
        let tensor = iota(&[2, 3, 4]);
        let borrowed = tensor.reshape(&[6, 4]).expect("size preserved");
        let consumed = tensor.into_reshape(&[6, 4]).expect("size preserved");
        assert_eq!(consumed.shape(), &[6, 4]);
        assert_eq!(consumed.to_vec(), borrowed.to_vec());
    }

    #[test]
    fn test_into_reshape_matches_reshape_non_contiguous() {
        // A non-contiguous source must be gathered in row-major logical order,
        // exactly like `reshape` does.
        let permuted = iota(&[2, 3, 4]).permute(&[2, 0, 1]).expect("valid");
        assert!(!permuted.is_contiguous());
        let borrowed = permuted.reshape(&[4, 6]).expect("size preserved");
        let consumed = permuted.into_reshape(&[4, 6]).expect("size preserved");
        assert_eq!(consumed.to_vec(), borrowed.to_vec());
    }

    #[test]
    fn test_into_reshape_rejects_size_mismatch() {
        let tensor = iota(&[2, 3]);
        let err = tensor
            .into_reshape(&[4, 4])
            .expect_err("element count differs");
        assert!(format!("{err}").contains("Cannot reshape"));
    }

    #[test]
    fn test_into_reshape_to_scalar_and_back() {
        let scalar = DenseND::<f64>::from_elem(&[1, 1, 1], 42.0);
        let zero_d = scalar.into_reshape(&[]).expect("1 element");
        assert_eq!(zero_d.rank(), 0);
        assert_eq!(zero_d.len(), 1);
        let back = zero_d.into_reshape(&[1]).expect("1 element");
        assert_eq!(back[&[0]], 42.0);
    }

    // -- permute_view / into_permuted -----------------------------------

    #[test]
    fn test_permute_view_matches_permute() {
        let tensor = iota(&[2, 3, 4]);
        let owned = tensor.permute(&[2, 0, 1]).expect("valid permutation");
        let view = tensor.permute_view(&[2, 0, 1]).expect("valid permutation");
        assert_eq!(view.shape(), owned.shape());
        assert_eq!(view, owned.as_array().view());
    }

    #[test]
    fn test_into_permuted_matches_permute() {
        let tensor = iota(&[2, 3, 4]);
        let owned = tensor.permute(&[1, 2, 0]).expect("valid permutation");
        let consumed = tensor.into_permuted(&[1, 2, 0]).expect("valid permutation");
        assert_eq!(consumed.shape(), owned.shape());
        assert_eq!(consumed.to_vec(), owned.to_vec());
        // Same (non-contiguous) layout, not just the same logical contents.
        assert_eq!(consumed.as_array().strides(), owned.as_array().strides());
    }

    #[test]
    fn test_permute_variants_reject_invalid_axes() {
        let tensor = iota(&[2, 3, 4]);
        // Wrong length
        assert!(tensor.permute(&[0, 1]).is_err());
        assert!(tensor.permute_view(&[0, 1]).is_err());
        assert!(tensor.clone().into_permuted(&[0, 1]).is_err());
        // Out of range
        assert!(tensor.permute(&[0, 1, 3]).is_err());
        assert!(tensor.permute_view(&[0, 1, 3]).is_err());
        assert!(tensor.clone().into_permuted(&[0, 1, 3]).is_err());
        // Duplicate
        assert!(tensor.permute(&[0, 1, 1]).is_err());
        assert!(tensor.permute_view(&[0, 1, 1]).is_err());
        assert!(tensor.clone().into_permuted(&[0, 1, 1]).is_err());
    }

    #[test]
    fn test_permute_validation_handles_rank_above_smallvec_inline_capacity() {
        // The `seen` scratch buffer has inline capacity 8; rank 10 spills to the
        // heap and must still validate correctly.
        let shape = [1_usize; 10];
        let tensor = DenseND::<f64>::ones(&shape);
        let identity: Vec<usize> = (0..10).collect();
        assert!(tensor.permute(&identity).is_ok());
        let mut duplicated = identity.clone();
        duplicated[9] = 0;
        assert!(tensor.permute(&duplicated).is_err());
    }

    // -- flatten on a non-contiguous tensor ------------------------------

    #[test]
    fn test_flatten_of_permuted_tensor_gathers_logical_order() {
        // Previously this panicked: `flatten` called `into_shape_with_order` on a
        // non-contiguous clone, which fails. It must gather instead.
        let tensor = iota(&[2, 3]);
        let permuted = tensor.permute(&[1, 0]).expect("valid permutation");
        let flat = permuted.flatten();
        assert_eq!(flat.shape(), &[6]);
        assert_eq!(flat.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

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
