//! Boolean mask interop methods for `DenseND<T>`.
//!
//! This module provides utilities for converting dense tensors to/from sparse
//! boolean-mask representations, enabling efficient sparse interop patterns.
//!
//! ## Overview
//!
//! - [`DenseND::to_bool_mask`]    — produce an `Array<bool, IxDyn>` marking non-zeros
//! - [`DenseND::apply_bool_mask`] — zero out elements where the mask is `false`
//! - [`DenseND::sparse_indices`]  — collect multi-dimensional indices of non-zero elements
//!
//! These operations are the bridge between the dense tensor world and the sparse
//! COO/CSR/CSF formats implemented in `tenrso-sparse`.

use super::types::DenseND;
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::Num;

impl<T> DenseND<T>
where
    T: Clone + Num + PartialOrd,
{
    /// Creates a boolean mask indicating where `|value| > threshold`.
    ///
    /// Returns an `Array<bool, IxDyn>` with the same shape as `self`.
    /// Element `mask[i]` is `true` when `abs(self[i]) > threshold`.
    ///
    /// Useful for converting a dense tensor to a sparse representation:
    /// the nonzero pattern can then be passed to [`apply_bool_mask`].
    ///
    /// # Example
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let t = DenseND::<f64>::from_vec(vec![0.0, 1.0, 0.0, -2.0], &[4]).unwrap();
    /// let mask = t.to_bool_mask(0.5);
    /// assert_eq!(mask[&[0usize][..]], false);
    /// assert_eq!(mask[&[1usize][..]], true);
    /// assert_eq!(mask[&[3usize][..]], true);
    /// ```
    pub fn to_bool_mask(&self, threshold: T) -> Array<bool, IxDyn> {
        self.data.mapv(|ref v| {
            // Compute absolute value without requiring the Signed trait:
            // abs(v) = if v >= 0 { v } else { 0 - v }
            let zero = T::zero();
            let abs_v = if *v >= zero {
                v.clone()
            } else {
                zero - v.clone()
            };
            abs_v > threshold.clone()
        })
    }

    /// Zeros out elements where `mask` is `false`.
    ///
    /// Returns a new tensor with `self[i]` where `mask[i]` is `true`,
    /// and `T::zero()` where `mask[i]` is `false`.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask.shape() != self.shape()`.
    ///
    /// # Example
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    /// use scirs2_core::ndarray_ext::{Array, IxDyn};
    ///
    /// let t = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let mask = Array::from_shape_vec(
    ///     IxDyn(&[2, 2]),
    ///     vec![true, false, false, true]
    /// ).unwrap();
    /// let masked = t.apply_bool_mask(&mask).unwrap();
    /// assert_eq!(masked[&[0, 0]], 1.0);
    /// assert_eq!(masked[&[0, 1]], 0.0);
    /// assert_eq!(masked[&[1, 1]], 4.0);
    /// ```
    pub fn apply_bool_mask(&self, mask: &Array<bool, IxDyn>) -> anyhow::Result<Self> {
        if mask.shape() != self.shape() {
            anyhow::bail!(
                "Mask shape {:?} does not match tensor shape {:?}",
                mask.shape(),
                self.shape()
            );
        }
        let new_data = scirs2_core::ndarray_ext::Zip::from(&self.data)
            .and(mask)
            .map_collect(|v, &m| if m { v.clone() } else { T::zero() });
        Ok(Self { data: new_data })
    }

    /// Returns the multi-dimensional indices of elements where `|value| > threshold`.
    ///
    /// Useful for building COO-format sparse tensors: each `Vec<usize>` in the
    /// returned list is a multi-index into the tensor where the element is
    /// considered non-zero.
    ///
    /// # Example
    ///
    /// ```
    /// use tenrso_core::dense::DenseND;
    ///
    /// let t = DenseND::<f64>::from_vec(vec![0.0, 1.0, 0.0, -2.0], &[4]).unwrap();
    /// let idxs = t.sparse_indices(0.5);
    /// assert_eq!(idxs.len(), 2);
    /// assert_eq!(idxs[0], vec![1usize]);
    /// assert_eq!(idxs[1], vec![3usize]);
    /// ```
    pub fn sparse_indices(&self, threshold: T) -> Vec<Vec<usize>> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        let mut result = Vec::new();

        if ndim == 0 {
            return result;
        }

        // Pre-compute row-major strides
        let mut strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        for (flat_idx, v) in self.data.iter().enumerate() {
            // Compute absolute value without requiring the Signed trait
            let zero = T::zero();
            let abs_v = if *v >= zero {
                v.clone()
            } else {
                zero - v.clone()
            };
            if abs_v > threshold.clone() {
                // Convert flat index to multi-dimensional index (row-major)
                let mut remaining = flat_idx;
                let mut multi_idx = vec![0usize; ndim];
                for d in 0..ndim {
                    multi_idx[d] = remaining / strides[d];
                    remaining %= strides[d];
                }
                result.push(multi_idx);
            }
        }
        result
    }
}

// ============================================================
// Tests for boolean mask interop
// ============================================================

#[cfg(test)]
mod mask_tests {
    use super::*;
    use scirs2_core::ndarray_ext::IxDyn;

    #[test]
    fn test_to_bool_mask_basic() {
        let t = DenseND::<f64>::from_vec(vec![0.0, 1.0, 0.0, -2.0], &[4]).unwrap();
        let mask = t.to_bool_mask(0.5);
        assert_eq!(mask.shape(), &[4]);
        assert!(!mask[&[0usize] as &[usize]]);
        assert!(mask[&[1usize] as &[usize]]);
        assert!(!mask[&[2usize] as &[usize]]);
        assert!(mask[&[3usize] as &[usize]]);
    }

    #[test]
    fn test_to_bool_mask_2d() {
        let t = DenseND::<f64>::from_vec(vec![0.0, 0.3, 1.0, -0.5, 2.0, 0.1], &[2, 3]).unwrap();
        let mask = t.to_bool_mask(0.5);
        assert_eq!(mask.shape(), &[2, 3]);
        // [0,0]=0.0 → false, [0,1]=0.3 → false, [0,2]=1.0 → true
        // [1,0]=-0.5 → false (|−0.5|=0.5, not > 0.5), [1,1]=2.0 → true, [1,2]=0.1 → false
        assert!(!mask[&[0usize, 0][..]]);
        assert!(!mask[&[0usize, 1][..]]);
        assert!(mask[&[0usize, 2][..]]);
        assert!(!mask[&[1usize, 0][..]]);
        assert!(mask[&[1usize, 1][..]]);
        assert!(!mask[&[1usize, 2][..]]);
    }

    #[test]
    fn test_to_bool_mask_shape_preserved() {
        let t = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
        let mask = t.to_bool_mask(0.5);
        assert_eq!(mask.shape(), t.shape());
    }

    #[test]
    fn test_apply_bool_mask_zeros_out() {
        let t = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mask = Array::from_shape_vec(IxDyn(&[2, 2]), vec![true, false, false, true]).unwrap();
        let masked = t.apply_bool_mask(&mask).unwrap();
        assert_eq!(masked[&[0, 0]], 1.0);
        assert_eq!(masked[&[0, 1]], 0.0);
        assert_eq!(masked[&[1, 0]], 0.0);
        assert_eq!(masked[&[1, 1]], 4.0);
    }

    #[test]
    fn test_apply_bool_mask_shape_mismatch() {
        let t = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mask = Array::from_shape_vec(IxDyn(&[4]), vec![true, false, true, false]).unwrap();
        let result = t.apply_bool_mask(&mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_indices_1d() {
        let t = DenseND::<f64>::from_vec(vec![0.0, 1.0, 0.0, -2.0], &[4]).unwrap();
        let idxs = t.sparse_indices(0.5);
        assert_eq!(idxs.len(), 2);
        assert_eq!(idxs[0], vec![1usize]);
        assert_eq!(idxs[1], vec![3usize]);
    }

    #[test]
    fn test_sparse_indices_2d() {
        let t = DenseND::<f64>::from_vec(vec![0.0, 1.0, 2.0, 0.0, 0.0, -3.0], &[2, 3]).unwrap();
        let idxs = t.sparse_indices(0.5);
        // Elements > 0.5 at flat indices 1, 2, 5
        // [0,1]=1.0, [0,2]=2.0, [1,2]=-3.0 (|-3|=3.0 > 0.5)
        assert_eq!(idxs.len(), 3);
        assert_eq!(idxs[0], vec![0usize, 1]);
        assert_eq!(idxs[1], vec![0usize, 2]);
        assert_eq!(idxs[2], vec![1usize, 2]);
    }

    #[test]
    fn test_sparse_indices_empty() {
        let t = DenseND::<f64>::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let idxs = t.sparse_indices(0.5);
        assert!(idxs.is_empty());
    }
}
