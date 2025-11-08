//! HiCOO (Hierarchical COO) format for N-dimensional sparse tensors
//!
//! # Overview
//!
//! HiCOO is a hierarchical blocked coordinate format that improves cache locality
//! for sparse tensor operations. It divides the tensor into blocks and stores
//! block coordinates separately from within-block coordinates.
//!
//! # Structure
//!
//! - `block_shape`: Size of each block dimension
//! - `block_coords`: COO-style coordinates of non-empty blocks
//! - `block_ptrs`: Pointers to start of each block's data
//! - `local_coords`: Within-block coordinates (relative to block start)
//! - `values`: Nonzero values
//!
//! # Example
//!
//! For a 3D tensor with block_shape=[2,2,2]:
//! - Tensor indices (0,1,2), (1,0,3), (2,3,4)
//! - Block indices: (0,0,1), (0,0,1), (1,1,2)
//! - Local coords: (0,1,0), (1,0,1), (0,1,0)
//!
//! # Complexity
//!
//! - **Construction from COO**: O(nnz × log(nnz)) for sorting + O(nnz) for grouping
//! - **Cache-blocked iteration**: O(nnz) with better cache locality
//! - **Memory**: O(nnz) + O(number of blocks)
//!
//! # Use Cases
//!
//! HiCOO is efficient for:
//! - Large sparse tensors with clustered nonzeros
//! - MTTKRP and other tensor operations
//! - Better cache performance than plain COO

use anyhow::{bail, Result};
use scirs2_core::numeric::Float;
use thiserror::Error;

use crate::coo::CooTensor;

#[derive(Error, Debug)]
pub enum HiCooError {
    #[error("Invalid block shape: {0}")]
    InvalidBlockShape(String),
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

/// HiCOO (Hierarchical COO) tensor
///
/// A blocked coordinate format that stores sparse tensors with improved cache locality.
///
/// # Type Parameters
///
/// - `T`: Element type (must implement `Float` trait from scirs2_core)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::{CooTensor, HiCooTensor};
///
/// // Create a 3D sparse tensor from COO
/// let mut coo = CooTensor::zeros(vec![8, 8, 8]).unwrap();
/// coo.push(vec![0, 1, 2], 5.0).unwrap();
/// coo.push(vec![1, 0, 3], 6.0).unwrap();
/// coo.push(vec![4, 5, 6], 7.0).unwrap();
///
/// // Convert to HiCOO with block size 4×4×4
/// let hicoo = HiCooTensor::from_coo(&coo, &[4, 4, 4]).unwrap();
/// assert_eq!(hicoo.nnz(), 3);
/// assert_eq!(hicoo.num_blocks(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct HiCooTensor<T> {
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Block shape (size of each block dimension)
    block_shape: Vec<usize>,
    /// Number of nonzeros
    nnz: usize,
    /// Block coordinates (COO-style)
    block_coords: Vec<Vec<usize>>,
    /// Pointers to start of each block's data (length = num_blocks + 1)
    block_ptrs: Vec<usize>,
    /// Local coordinates within each block
    local_coords: Vec<Vec<usize>>,
    /// Values
    values: Vec<T>,
}

impl<T: Float> HiCooTensor<T> {
    /// Creates a HiCOO tensor from COO format with the specified block shape.
    ///
    /// # Arguments
    ///
    /// - `coo`: Input tensor in COO format
    /// - `block_shape`: Size of each block dimension
    ///
    /// # Complexity
    ///
    /// O(nnz × log(nnz)) for sorting + O(nnz) for grouping into blocks
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, HiCooTensor};
    ///
    /// let mut coo = CooTensor::zeros(vec![8, 8, 8]).unwrap();
    /// coo.push(vec![0, 1, 2], 1.0).unwrap();
    /// coo.push(vec![1, 2, 3], 2.0).unwrap();
    ///
    /// // Convert with 2×2×2 blocks
    /// let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
    /// assert_eq!(hicoo.nnz(), 2);
    /// ```
    pub fn from_coo(coo: &CooTensor<T>, block_shape: &[usize]) -> Result<Self> {
        let ndim = coo.shape().len();

        // Validate block shape
        if block_shape.len() != ndim {
            bail!(HiCooError::InvalidBlockShape(format!(
                "block_shape length {} != ndim {}",
                block_shape.len(),
                ndim
            )));
        }

        if block_shape.contains(&0) {
            bail!(HiCooError::InvalidBlockShape(
                "block_shape cannot contain zeros".to_string()
            ));
        }

        if coo.nnz() == 0 {
            return Ok(Self {
                shape: coo.shape().to_vec(),
                block_shape: block_shape.to_vec(),
                nnz: 0,
                block_coords: vec![],
                block_ptrs: vec![0],
                local_coords: vec![],
                values: vec![],
            });
        }

        // Build (block_coord, local_coord, value) tuples
        let mut block_data: Vec<(Vec<usize>, Vec<usize>, T)> = coo
            .indices()
            .iter()
            .zip(coo.values().iter())
            .map(|(idx, &val)| {
                let block_coord: Vec<usize> = idx
                    .iter()
                    .zip(block_shape)
                    .map(|(&i, &bs)| i / bs)
                    .collect();
                let local_coord: Vec<usize> = idx
                    .iter()
                    .zip(block_shape)
                    .map(|(&i, &bs)| i % bs)
                    .collect();
                (block_coord, local_coord, val)
            })
            .collect();

        // Sort by block coordinates
        block_data.sort_by(|a, b| a.0.cmp(&b.0));

        // Extract sorted data
        let mut block_coords = Vec::new();
        let mut block_ptrs = vec![0];
        let mut local_coords = Vec::new();
        let mut values = Vec::new();

        let mut current_block: Option<Vec<usize>> = None;
        for (block_coord, local_coord, value) in block_data {
            if current_block.as_ref() != Some(&block_coord) {
                // New block
                if current_block.is_some() {
                    // Not the first block, so record where the previous block ended
                    block_ptrs.push(values.len());
                }
                block_coords.push(block_coord.clone());
                current_block = Some(block_coord);
            }
            local_coords.push(local_coord);
            values.push(value);
        }
        // Final sentinel pointer
        block_ptrs.push(values.len());

        Ok(Self {
            shape: coo.shape().to_vec(),
            block_shape: block_shape.to_vec(),
            nnz: values.len(),
            block_coords,
            block_ptrs,
            local_coords,
            values,
        })
    }

    /// Returns the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the block shape.
    #[inline]
    pub fn block_shape(&self) -> &[usize] {
        &self.block_shape
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

    /// Returns the number of non-empty blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.block_coords.len()
    }

    /// Returns the density (nnz / total elements).
    pub fn density(&self) -> f64 {
        let total: usize = self.shape.iter().product();
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Returns a reference to the block coordinates.
    #[inline]
    pub fn block_coords(&self) -> &[Vec<usize>] {
        &self.block_coords
    }

    /// Returns a reference to the block pointers.
    #[inline]
    pub fn block_ptrs(&self) -> &[usize] {
        &self.block_ptrs
    }

    /// Returns a reference to the local coordinates.
    #[inline]
    pub fn local_coords(&self) -> &[Vec<usize>] {
        &self.local_coords
    }

    /// Returns a reference to the values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
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
    /// use tenrso_sparse::{CooTensor, HiCooTensor};
    ///
    /// let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
    /// coo.push(vec![0, 0, 1], 1.0).unwrap();
    /// coo.push(vec![2, 2, 2], 2.0).unwrap();
    ///
    /// let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
    /// let nonzeros: Vec<_> = hicoo.iter().collect();
    /// assert_eq!(nonzeros.len(), 2);
    /// ```
    pub fn iter(&self) -> HiCooIterator<'_, T> {
        HiCooIterator::new(self)
    }

    /// Converts HiCOO back to COO format.
    ///
    /// # Complexity
    ///
    /// O(nnz)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, HiCooTensor};
    ///
    /// let mut coo = CooTensor::zeros(vec![8, 8, 8]).unwrap();
    /// coo.push(vec![0, 1, 2], 5.0).unwrap();
    /// coo.push(vec![4, 5, 6], 7.0).unwrap();
    ///
    /// let hicoo = HiCooTensor::from_coo(&coo, &[4, 4, 4]).unwrap();
    /// let coo_back = hicoo.to_coo().unwrap();
    /// assert_eq!(coo_back.nnz(), 2);
    /// ```
    pub fn to_coo(&self) -> Result<CooTensor<T>> {
        let mut coo = CooTensor::zeros(self.shape.to_vec())?;

        for (indices, value) in self.iter() {
            coo.push(indices, value)?;
        }

        Ok(coo)
    }

    /// Converts HiCOO to dense tensor.
    ///
    /// # Complexity
    ///
    /// O(nnz + total_elements)
    ///
    /// # Examples
    ///
    /// ```
    /// use tenrso_sparse::{CooTensor, HiCooTensor};
    /// use scirs2_core::ndarray_ext::Array;
    ///
    /// let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
    /// coo.push(vec![0, 0, 0], 1.0).unwrap();
    /// coo.push(vec![2, 2, 2], 2.0).unwrap();
    ///
    /// let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
    /// let dense = hicoo.to_dense().unwrap();
    /// assert_eq!(dense[[0, 0, 0]], 1.0);
    /// assert_eq!(dense[[2, 2, 2]], 2.0);
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

/// Iterator over HiCOO tensor nonzeros
///
/// Yields (indices, value) tuples in block-major order (better cache locality).
pub struct HiCooIterator<'a, T> {
    hicoo: &'a HiCooTensor<T>,
    /// Current value index
    val_idx: usize,
}

impl<'a, T: Float> HiCooIterator<'a, T> {
    fn new(hicoo: &'a HiCooTensor<T>) -> Self {
        Self { hicoo, val_idx: 0 }
    }
}

impl<'a, T: Float> Iterator for HiCooIterator<'a, T> {
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.val_idx >= self.hicoo.nnz() {
            return None;
        }

        // Find which block contains this value using binary search
        let mut block_idx = 0;
        for i in 0..self.hicoo.num_blocks() {
            if self.val_idx >= self.hicoo.block_ptrs[i]
                && self.val_idx < self.hicoo.block_ptrs[i + 1]
            {
                block_idx = i;
                break;
            }
        }

        // Get block and local coordinates
        let block_coord = &self.hicoo.block_coords[block_idx];
        let local_coord = &self.hicoo.local_coords[self.val_idx];

        // Compute global indices
        let indices: Vec<usize> = block_coord
            .iter()
            .zip(local_coord)
            .zip(&self.hicoo.block_shape)
            .map(|((&bc, &lc), &bs)| bc * bs + lc)
            .collect();

        let value = self.hicoo.values[self.val_idx];

        self.val_idx += 1;
        Some((indices, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hicoo_from_coo_basic() {
        let mut coo = CooTensor::zeros(vec![8, 8, 8]).unwrap();
        coo.push(vec![0, 1, 2], 5.0).unwrap();
        coo.push(vec![1, 0, 3], 6.0).unwrap();
        coo.push(vec![4, 5, 6], 7.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[4, 4, 4]).unwrap();

        assert_eq!(hicoo.shape(), &[8, 8, 8]);
        assert_eq!(hicoo.nnz(), 3);
        assert_eq!(hicoo.num_blocks(), 2); // Blocks (0,0,0) and (1,1,1)
    }

    #[test]
    fn test_hicoo_invalid_block_shape() {
        let coo = CooTensor::<f64>::zeros(vec![8, 8, 8]).unwrap();

        // Wrong length
        let result = HiCooTensor::from_coo(&coo, &[4, 4]);
        assert!(result.is_err());

        // Contains zero
        let result = HiCooTensor::from_coo(&coo, &[4, 0, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hicoo_empty_tensor() {
        let coo = CooTensor::<f64>::zeros(vec![8, 8, 8]).unwrap();
        let hicoo = HiCooTensor::from_coo(&coo, &[4, 4, 4]).unwrap();

        assert_eq!(hicoo.nnz(), 0);
        assert_eq!(hicoo.num_blocks(), 0);
    }

    #[test]
    fn test_hicoo_iteration() {
        let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
        coo.push(vec![0, 0, 1], 1.0).unwrap();
        coo.push(vec![2, 2, 2], 2.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let nonzeros: Vec<_> = hicoo.iter().collect();

        assert_eq!(nonzeros.len(), 2);
        assert!(nonzeros.contains(&(vec![0, 0, 1], 1.0)));
        assert!(nonzeros.contains(&(vec![2, 2, 2], 2.0)));
    }

    #[test]
    fn test_hicoo_to_coo_roundtrip() {
        let mut coo = CooTensor::zeros(vec![8, 8, 8]).unwrap();
        coo.push(vec![0, 1, 2], 5.0).unwrap();
        coo.push(vec![4, 5, 6], 7.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[4, 4, 4]).unwrap();
        let coo_back = hicoo.to_coo().unwrap();

        assert_eq!(coo_back.nnz(), 2);
        assert_eq!(coo_back.shape(), &[8, 8, 8]);

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
    fn test_hicoo_to_dense() {
        let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
        coo.push(vec![0, 0, 0], 1.0).unwrap();
        coo.push(vec![2, 2, 2], 2.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        let dense = hicoo.to_dense().unwrap();

        assert_eq!(dense[[0, 0, 0]], 1.0);
        assert_eq!(dense[[2, 2, 2]], 2.0);
        assert_eq!(dense[[0, 1, 0]], 0.0);
    }

    #[test]
    fn test_hicoo_density() {
        let mut coo = CooTensor::zeros(vec![10, 10, 10]).unwrap();
        coo.push(vec![0, 0, 0], 1.0).unwrap();
        coo.push(vec![5, 5, 5], 2.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[5, 5, 5]).unwrap();
        let density = hicoo.density();

        assert!((density - 0.002).abs() < 1e-6); // 2 / 1000 = 0.002
    }

    #[test]
    fn test_hicoo_single_element() {
        let mut coo = CooTensor::zeros(vec![8, 8, 8]).unwrap();
        coo.push(vec![2, 3, 4], 42.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[4, 4, 4]).unwrap();
        assert_eq!(hicoo.nnz(), 1);
        assert_eq!(hicoo.num_blocks(), 1);

        let nonzeros: Vec<_> = hicoo.iter().collect();
        assert_eq!(nonzeros.len(), 1);
        assert_eq!(nonzeros[0], (vec![2, 3, 4], 42.0));
    }

    #[test]
    fn test_hicoo_block_grouping() {
        let mut coo = CooTensor::zeros(vec![4, 4, 4]).unwrap();
        // All in same block (0,0,0) with block_shape [2,2,2]
        coo.push(vec![0, 0, 0], 1.0).unwrap();
        coo.push(vec![0, 1, 1], 2.0).unwrap();
        coo.push(vec![1, 0, 1], 3.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2]).unwrap();
        assert_eq!(hicoo.nnz(), 3);
        assert_eq!(hicoo.num_blocks(), 1); // All in same block

        // Different blocks
        let mut coo2 = CooTensor::zeros(vec![4, 4, 4]).unwrap();
        coo2.push(vec![0, 0, 0], 1.0).unwrap();
        coo2.push(vec![2, 2, 2], 2.0).unwrap();

        let hicoo2 = HiCooTensor::from_coo(&coo2, &[2, 2, 2]).unwrap();
        assert_eq!(hicoo2.nnz(), 2);
        assert_eq!(hicoo2.num_blocks(), 2); // In different blocks
    }

    #[test]
    fn test_hicoo_high_dimensional() {
        let mut coo = CooTensor::zeros(vec![4, 4, 4, 4, 4]).unwrap();
        coo.push(vec![0, 0, 0, 0, 1], 1.0).unwrap();
        coo.push(vec![2, 2, 2, 2, 2], 2.0).unwrap();

        let hicoo = HiCooTensor::from_coo(&coo, &[2, 2, 2, 2, 2]).unwrap();
        assert_eq!(hicoo.nnz(), 2);
        assert_eq!(hicoo.ndim(), 5);

        let dense = hicoo.to_dense().unwrap();
        assert_eq!(dense[[0, 0, 0, 0, 1]], 1.0);
        assert_eq!(dense[[2, 2, 2, 2, 2]], 2.0);
    }
}
