//! CSF (Compressed Sparse Fiber) format for N-dimensional sparse tensors
//!
//! # Overview
//!
//! CSF is a hierarchical sparse tensor format that generalizes CSR/CSC to N dimensions.
//! It stores tensors as a tree of compressed fibers, where each level represents one mode.
//!
//! # Structure
//!
//! For an n-mode tensor, CSF has:
//! - `mode_order`: Permutation of modes (e.g., [0,1,2] or [2,1,0])
//! - `fptr[i]`: Fiber pointers at level i (i = 0..n-1)
//! - `fids[i]`: Fiber indices at level i (i = 0..n-1)
//! - `vals`: Nonzero values at leaf level
//!
//! # Example
//!
//! For a 3D tensor with nonzeros at (0,1,2)=5, (0,1,3)=6, (1,2,3)=7:
//!
//! Mode order [0,1,2]:
//! - Level 0 (mode 0):
//!   - fptr[0] = [0, 1, 2]  // 2 unique mode-0 indices (0 and 1)
//!   - fids[0] = [0, 1]      // mode-0 indices
//! - Level 1 (mode 1):
//!   - fptr[1] = [0, 1, 2]  // fibers for each mode-0 index
//!   - fids[1] = [1, 2]      // mode-1 indices
//! - Level 2 (mode 2, leaf):
//!   - fptr[2] = [0, 2, 3]  // values for each (mode-0, mode-1) pair
//!   - fids[2] = [2, 3, 3]   // mode-2 indices
//!   - vals = [5.0, 6.0, 7.0]
//!
//! # Complexity
//!
//! - **Construction from COO**: O(nnz × log(nnz)) for sorting + O(nnz) for building tree
//! - **Fiber iteration**: O(nnz) to visit all nonzeros
//! - **Memory**: O(nnz) + O(number of fibers at each level)
//!
//! # Use Cases
//!
//! CSF is efficient for:
//! - High-dimensional sparse tensors (≥3 modes)
//! - MTTKRP (Matricized Tensor Times Khatri-Rao Product)
//! - Tensor contractions with specific mode ordering
//! - Hierarchical iteration patterns

use anyhow::{bail, Result};
use scirs2_core::numeric::Float;
use thiserror::Error;

use crate::coo::CooTensor;

#[derive(Error, Debug)]
pub enum CsfError {
    #[error("Invalid mode order: {0}")]
    InvalidModeOrder(String),
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Empty tensor")]
    EmptyTensor,
}

/// CSF (Compressed Sparse Fiber) tensor
///
/// A hierarchical sparse tensor format that organizes nonzeros into a tree of fibers.
/// Each level of the tree corresponds to one mode of the tensor.
///
/// # Type Parameters
///
/// - `T`: Element type (must implement `Float` trait from scirs2_core)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::CsfTensor;
/// use tenrso_sparse::CooTensor;
///
/// // Create a 3D sparse tensor from COO
/// let mut coo = CooTensor::zeros(vec![3, 4, 5]).unwrap();
/// coo.push(vec![0, 1, 2], 5.0).unwrap();
/// coo.push(vec![0, 1, 3], 6.0).unwrap();
/// coo.push(vec![1, 2, 3], 7.0).unwrap();
///
/// // Convert to CSF with mode order [0, 1, 2]
/// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
/// assert_eq!(csf.nnz(), 3);
/// assert_eq!(csf.shape(), &[3, 4, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct CsfTensor<T> {
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Mode ordering (permutation of 0..ndim)
    mode_order: Vec<usize>,
    /// Number of nonzeros
    nnz: usize,
    /// Fiber pointers for each level
    /// fptr[i] has length = len(fids[i]) + 1
    fptr: Vec<Vec<usize>>,
    /// Fiber indices for each level
    fids: Vec<Vec<usize>>,
    /// Values (at leaf level)
    vals: Vec<T>,
}

impl<T: Float> CsfTensor<T> {
    /// Creates a CSF tensor from COO format with the specified mode ordering.
    ///
    /// # Arguments
    ///
    /// - `coo`: Input tensor in COO format
    /// - `mode_order`: Permutation of modes (e.g., [0,1,2] or [2,1,0])
    ///
    /// # Complexity
    ///
    /// O(nnz × log(nnz)) for sorting + O(nnz) for tree construction
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, CsfTensor};
    ///
    /// let mut coo = CooTensor::zeros(vec![3, 3, 3]).unwrap();
    /// coo.push(vec![0, 1, 2], 1.0).unwrap();
    /// coo.push(vec![1, 2, 0], 2.0).unwrap();
    ///
    /// // Convert with natural mode order
    /// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
    /// assert_eq!(csf.nnz(), 2);
    ///
    /// // Convert with reversed mode order
    /// let csf_rev = CsfTensor::from_coo(&coo, &[2, 1, 0]).unwrap();
    /// assert_eq!(csf_rev.nnz(), 2);
    /// ```
    pub fn from_coo(coo: &CooTensor<T>, mode_order: &[usize]) -> Result<Self> {
        let ndim = coo.shape().len();

        // Validate mode order
        if mode_order.len() != ndim {
            bail!(CsfError::InvalidModeOrder(format!(
                "mode_order length {} != ndim {}",
                mode_order.len(),
                ndim
            )));
        }

        let mut sorted_modes = mode_order.to_vec();
        sorted_modes.sort_unstable();
        if sorted_modes != (0..ndim).collect::<Vec<_>>() {
            bail!(CsfError::InvalidModeOrder(format!(
                "mode_order {:?} is not a permutation of 0..{}",
                mode_order, ndim
            )));
        }

        if coo.nnz() == 0 {
            // Return empty CSF tensor
            return Ok(Self {
                shape: coo.shape().to_vec(),
                mode_order: mode_order.to_vec(),
                nnz: 0,
                fptr: vec![vec![0]; ndim],
                fids: vec![vec![]; ndim],
                vals: vec![],
            });
        }

        // Sort COO by the specified mode order
        let mut sorted_indices: Vec<(Vec<usize>, T)> = coo
            .indices()
            .iter()
            .zip(coo.values().iter())
            .map(|(idx, &val)| (idx.clone(), val))
            .collect();

        sorted_indices.sort_by(|a, b| {
            for &mode in mode_order {
                match a.0[mode].cmp(&b.0[mode]) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            std::cmp::Ordering::Equal
        });

        // Build CSF tree level by level
        let mut fptr = Vec::with_capacity(ndim);
        let mut fids = Vec::with_capacity(ndim);
        let mut vals = Vec::with_capacity(sorted_indices.len());

        // Level 0: root level
        let mut current_ptrs = vec![0];
        let mut current_indices = Vec::new();

        // Group by first mode
        let first_mode = mode_order[0];
        let mut i = 0;
        while i < sorted_indices.len() {
            let idx_val = sorted_indices[i].0[first_mode];
            current_indices.push(idx_val);

            // Count how many elements have the same first index
            let mut j = i;
            while j < sorted_indices.len() && sorted_indices[j].0[first_mode] == idx_val {
                j += 1;
            }
            current_ptrs.push(j - i + current_ptrs.last().unwrap());
            i = j;
        }

        fptr.push(current_ptrs);
        fids.push(current_indices);

        // Build remaining levels
        for level in 1..ndim {
            let mode = mode_order[level];
            let mut next_ptrs = vec![0];
            let mut next_indices = Vec::new();

            // For each fiber at previous level
            let prev_fptr = &fptr[level - 1];
            for fiber_idx in 0..prev_fptr.len() - 1 {
                let start = prev_fptr[fiber_idx];
                let end = prev_fptr[fiber_idx + 1];

                if level == ndim - 1 {
                    // Leaf level: extract values and indices
                    for (indices, value) in sorted_indices.iter().skip(start).take(end - start) {
                        next_indices.push(indices[mode]);
                        vals.push(*value);
                    }
                    next_ptrs.push(vals.len());
                } else {
                    // Non-leaf level: group by current mode
                    let mut local_start = start;
                    while local_start < end {
                        let idx_val = sorted_indices[local_start].0[mode];
                        next_indices.push(idx_val);

                        let mut local_end = local_start;
                        while local_end < end && sorted_indices[local_end].0[mode] == idx_val {
                            local_end += 1;
                        }
                        next_ptrs.push(next_ptrs.last().unwrap() + (local_end - local_start));
                        local_start = local_end;
                    }
                }
            }

            fptr.push(next_ptrs);
            fids.push(next_indices);
        }

        Ok(Self {
            shape: coo.shape().to_vec(),
            mode_order: mode_order.to_vec(),
            nnz: sorted_indices.len(),
            fptr,
            fids,
            vals,
        })
    }

    /// Returns the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the number of nonzeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Returns the mode ordering.
    #[inline]
    pub fn mode_order(&self) -> &[usize] {
        &self.mode_order
    }

    /// Returns the density (nnz / total elements).
    pub fn density(&self) -> f64 {
        let total: usize = self.shape.iter().product();
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Returns a reference to the fiber pointers at the specified level.
    ///
    /// # Arguments
    ///
    /// - `level`: Level index (0 to ndim-1)
    ///
    /// # Panics
    ///
    /// Panics if level >= ndim.
    #[inline]
    pub fn fptr(&self, level: usize) -> &[usize] {
        &self.fptr[level]
    }

    /// Returns a reference to the fiber indices at the specified level.
    ///
    /// # Arguments
    ///
    /// - `level`: Level index (0 to ndim-1)
    ///
    /// # Panics
    ///
    /// Panics if level >= ndim.
    #[inline]
    pub fn fids(&self, level: usize) -> &[usize] {
        &self.fids[level]
    }

    /// Returns a reference to the values.
    #[inline]
    pub fn vals(&self) -> &[T] {
        &self.vals
    }

    /// Iterates over all nonzeros as (indices, value) tuples.
    ///
    /// # Complexity
    ///
    /// O(nnz)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, CsfTensor};
    ///
    /// let mut coo = CooTensor::zeros(vec![2, 2, 2]).unwrap();
    /// coo.push(vec![0, 0, 1], 1.0).unwrap();
    /// coo.push(vec![1, 1, 0], 2.0).unwrap();
    ///
    /// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
    /// let nonzeros: Vec<_> = csf.iter().collect();
    /// assert_eq!(nonzeros.len(), 2);
    /// ```
    pub fn iter(&self) -> CsfIterator<'_, T> {
        CsfIterator::new(self)
    }

    /// Converts CSF back to COO format.
    ///
    /// # Complexity
    ///
    /// O(nnz)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, CsfTensor};
    ///
    /// let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
    /// coo.push(vec![0, 1, 2], 5.0).unwrap();
    /// coo.push(vec![1, 2, 3], 7.0).unwrap();
    ///
    /// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
    /// let coo_back = csf.to_coo().unwrap();
    /// assert_eq!(coo_back.nnz(), 2);
    /// ```
    pub fn to_coo(&self) -> Result<CooTensor<T>> {
        let mut coo = CooTensor::zeros(self.shape.to_vec())?;

        for (indices, value) in self.iter() {
            coo.push(indices, value)?;
        }

        Ok(coo)
    }

    /// Converts CSF to dense tensor.
    ///
    /// # Complexity
    ///
    /// O(nnz + total_elements)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, CsfTensor};
    /// use scirs2_core::ndarray_ext::Array;
    ///
    /// let mut coo = CooTensor::zeros(vec![2, 2, 2]).unwrap();
    /// coo.push(vec![0, 0, 0], 1.0).unwrap();
    /// coo.push(vec![1, 1, 1], 2.0).unwrap();
    ///
    /// let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
    /// let dense = csf.to_dense().unwrap();
    /// assert_eq!(dense[[0, 0, 0]], 1.0);
    /// assert_eq!(dense[[1, 1, 1]], 2.0);
    /// assert_eq!(dense[[0, 1, 0]], 0.0);
    /// ```
    pub fn to_dense(&self) -> Result<scirs2_core::ndarray_ext::ArrayD<T>> {
        use scirs2_core::ndarray_ext::ArrayD;

        let mut dense = ArrayD::zeros(self.shape.to_vec());

        for (indices, value) in self.iter() {
            dense[indices.as_slice()] = value;
        }

        Ok(dense)
    }
}

/// Iterator over CSF tensor nonzeros
///
/// Yields (indices, value) tuples in the order determined by the CSF mode ordering.
pub struct CsfIterator<'a, T> {
    csf: &'a CsfTensor<T>,
    /// Current value index
    val_idx: usize,
}

impl<'a, T: Float> CsfIterator<'a, T> {
    fn new(csf: &'a CsfTensor<T>) -> Self {
        Self { csf, val_idx: 0 }
    }
}

impl<'a, T: Float> Iterator for CsfIterator<'a, T> {
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.val_idx >= self.csf.nnz() {
            return None;
        }

        let ndim = self.csf.ndim();
        let mut indices = vec![0; ndim];
        let value = self.csf.vals[self.val_idx];

        // Reconstruct indices from CSF structure
        // We need to find which fiber path leads to val_idx
        let mut current_val_pos = self.val_idx;

        // Work backwards from leaf level to root
        for level in (0..ndim).rev() {
            if level == ndim - 1 {
                // Leaf level: find which fiber contains this value
                let mut fiber_idx = 0;
                for i in 0..self.csf.fptr[level].len() - 1 {
                    let start = self.csf.fptr[level][i];
                    let end = self.csf.fptr[level][i + 1];
                    if current_val_pos >= start && current_val_pos < end {
                        fiber_idx = i;
                        break;
                    }
                }

                // Get the index at this level
                let mode = self.csf.mode_order[level];
                indices[mode] = self.csf.fids[level][current_val_pos];

                // Update position for next level up
                current_val_pos = fiber_idx;
            } else {
                // Non-leaf level: find which fiber contains current_val_pos
                let mut fiber_idx = 0;
                for i in 0..self.csf.fptr[level].len() - 1 {
                    let start = self.csf.fptr[level][i];
                    let end = self.csf.fptr[level][i + 1];
                    if current_val_pos >= start && current_val_pos < end {
                        fiber_idx = i;
                        break;
                    }
                }

                // Get the index at this level
                let mode = self.csf.mode_order[level];
                indices[mode] = self.csf.fids[level][current_val_pos];

                // Update position for next level up
                current_val_pos = fiber_idx;
            }
        }

        self.val_idx += 1;
        Some((indices, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csf_from_coo_basic() {
        let mut coo = CooTensor::zeros(vec![3, 4, 5]).unwrap();
        coo.push(vec![0, 1, 2], 5.0).unwrap();
        coo.push(vec![0, 1, 3], 6.0).unwrap();
        coo.push(vec![1, 2, 3], 7.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        assert_eq!(csf.shape(), &[3, 4, 5]);
        assert_eq!(csf.nnz(), 3);
        assert_eq!(csf.mode_order(), &[0, 1, 2]);
    }

    #[test]
    fn test_csf_invalid_mode_order() {
        let coo = CooTensor::<f64>::zeros(vec![3, 3, 3]).unwrap();

        // Wrong length
        let result = CsfTensor::from_coo(&coo, &[0, 1]);
        assert!(result.is_err());

        // Not a permutation
        let result = CsfTensor::from_coo(&coo, &[0, 0, 1]);
        assert!(result.is_err());

        // Out of range
        let result = CsfTensor::from_coo(&coo, &[0, 1, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_csf_empty_tensor() {
        let coo = CooTensor::<f64>::zeros(vec![3, 3, 3]).unwrap();
        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        assert_eq!(csf.nnz(), 0);
        assert_eq!(csf.shape(), &[3, 3, 3]);
    }

    #[test]
    fn test_csf_different_mode_orders() {
        let mut coo = CooTensor::zeros(vec![2, 3, 4]).unwrap();
        coo.push(vec![0, 1, 2], 1.0).unwrap();
        coo.push(vec![1, 2, 3], 2.0).unwrap();

        // Natural order
        let csf1 = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        assert_eq!(csf1.nnz(), 2);

        // Reversed order
        let csf2 = CsfTensor::from_coo(&coo, &[2, 1, 0]).unwrap();
        assert_eq!(csf2.nnz(), 2);

        // Custom order
        let csf3 = CsfTensor::from_coo(&coo, &[1, 0, 2]).unwrap();
        assert_eq!(csf3.nnz(), 2);
    }

    #[test]
    fn test_csf_iteration() {
        let mut coo = CooTensor::zeros(vec![2, 2, 2]).unwrap();
        coo.push(vec![0, 0, 1], 1.0).unwrap();
        coo.push(vec![1, 1, 0], 2.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let nonzeros: Vec<_> = csf.iter().collect();

        assert_eq!(nonzeros.len(), 2);
        assert!(nonzeros.contains(&(vec![0, 0, 1], 1.0)));
        assert!(nonzeros.contains(&(vec![1, 1, 0], 2.0)));
    }

    #[test]
    fn test_csf_to_coo_roundtrip() {
        let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
        coo.push(vec![0, 1, 2], 5.0).unwrap();
        coo.push(vec![1, 2, 3], 7.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let coo_back = csf.to_coo().unwrap();

        assert_eq!(coo_back.nnz(), 2);
        assert_eq!(coo_back.shape(), &[4, 4, 4]);

        // Check values are preserved by sorting and comparing
        let mut orig_pairs: Vec<_> = coo
            .indices()
            .iter()
            .zip(coo.values())
            .map(|(idx, &val)| (idx.clone(), val))
            .collect();
        orig_pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let mut back_pairs: Vec<_> = coo_back
            .indices()
            .iter()
            .zip(coo_back.values())
            .map(|(idx, &val)| (idx.clone(), val))
            .collect();
        back_pairs.sort_by(|a, b| a.0.cmp(&b.0));

        for (orig, back) in orig_pairs.iter().zip(back_pairs.iter()) {
            assert_eq!(orig.0, back.0);
            assert!((orig.1 - back.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_csf_to_dense() {
        let mut coo = CooTensor::zeros(vec![2, 2, 2]).unwrap();
        coo.push(vec![0, 0, 0], 1.0).unwrap();
        coo.push(vec![1, 1, 1], 2.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let dense = csf.to_dense().unwrap();

        assert_eq!(dense[[0, 0, 0]], 1.0);
        assert_eq!(dense[[1, 1, 1]], 2.0);
        assert_eq!(dense[[0, 1, 0]], 0.0);
        assert_eq!(dense[[1, 0, 1]], 0.0);
    }

    #[test]
    fn test_csf_density() {
        let mut coo = CooTensor::zeros(vec![10, 10, 10]).unwrap();
        coo.push(vec![0, 0, 0], 1.0).unwrap();
        coo.push(vec![5, 5, 5], 2.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        let density = csf.density();

        assert!((density - 0.002).abs() < 1e-6); // 2 / 1000 = 0.002
    }

    #[test]
    fn test_csf_single_element() {
        let mut coo = CooTensor::zeros(vec![5, 5, 5]).unwrap();
        coo.push(vec![2, 3, 4], 42.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();
        assert_eq!(csf.nnz(), 1);

        let nonzeros: Vec<_> = csf.iter().collect();
        assert_eq!(nonzeros.len(), 1);
        assert_eq!(nonzeros[0], (vec![2, 3, 4], 42.0));
    }

    #[test]
    fn test_csf_fiber_access() {
        let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
        coo.push(vec![0, 1, 2], 1.0).unwrap();
        coo.push(vec![0, 1, 3], 2.0).unwrap();
        coo.push(vec![1, 2, 3], 3.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2]).unwrap();

        // Check level 0 (mode 0)
        let fptr0 = csf.fptr(0);
        let fids0 = csf.fids(0);
        assert!(!fptr0.is_empty());
        assert!(!fids0.is_empty());

        // Check values
        let vals = csf.vals();
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_csf_high_dimensional() {
        let mut coo = CooTensor::zeros(vec![2, 2, 2, 2, 2]).unwrap();
        coo.push(vec![0, 0, 0, 0, 1], 1.0).unwrap();
        coo.push(vec![1, 1, 1, 1, 0], 2.0).unwrap();

        let csf = CsfTensor::from_coo(&coo, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(csf.nnz(), 2);
        assert_eq!(csf.ndim(), 5);

        let dense = csf.to_dense().unwrap();
        assert_eq!(dense[[0, 0, 0, 0, 1]], 1.0);
        assert_eq!(dense[[1, 1, 1, 1, 0]], 2.0);
    }
}
