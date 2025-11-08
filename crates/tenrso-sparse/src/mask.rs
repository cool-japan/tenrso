//! Masking support for sparse tensor operations
//!
//! Provides efficient boolean masks for selective computation in tensor operations.
//!
//! # Purpose
//!
//! Masked operations allow computing only at specified locations, which is essential for:
//! - Sparse-dense mixed operations
//! - Selective output computation
//! - Planner-guided execution
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::mask::Mask;
//!
//! // Create a mask for a 3×4 tensor
//! let indices = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
//! let mask = Mask::from_indices(indices, vec![3, 4]).unwrap();
//!
//! assert_eq!(mask.nnz(), 3);
//! assert!(mask.contains(&[0, 1]));
//! assert!(!mask.contains(&[0, 0]));
//! ```

use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MaskError {
    #[error("Index {index:?} out of bounds for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Boolean mask for selective tensor operations
///
/// Represents a sparse set of indices where operations should be performed.
#[derive(Debug, Clone)]
pub struct Mask {
    /// Sparse set of active indices
    indices: HashSet<Vec<usize>>,

    /// Shape of the masked tensor
    shape: Vec<usize>,
}

impl Mask {
    /// Create a mask from a list of indices
    ///
    /// # Arguments
    ///
    /// * `indices` - List of multi-dimensional indices to include in mask
    /// * `shape` - Shape of the tensor being masked
    ///
    /// # Errors
    ///
    /// Returns error if any index is out of bounds or shape is invalid.
    pub fn from_indices(indices: Vec<Vec<usize>>, shape: Vec<usize>) -> Result<Self, MaskError> {
        // Validate shape
        if shape.is_empty() || shape.contains(&0) {
            return Err(MaskError::InvalidShape(
                "Shape cannot be empty or contain zeros".to_string(),
            ));
        }

        // Validate all indices
        for idx in &indices {
            if idx.len() != shape.len() {
                return Err(MaskError::ShapeMismatch {
                    expected: shape.clone(),
                    actual: vec![idx.len()],
                });
            }

            for (i, &coord) in idx.iter().enumerate() {
                if coord >= shape[i] {
                    return Err(MaskError::IndexOutOfBounds {
                        index: idx.clone(),
                        shape: shape.clone(),
                    });
                }
            }
        }

        Ok(Self {
            indices: indices.into_iter().collect(),
            shape,
        })
    }

    /// Create an empty mask
    pub fn empty(shape: Vec<usize>) -> Result<Self, MaskError> {
        Self::from_indices(Vec::new(), shape)
    }

    /// Create a full mask (all indices active)
    pub fn full(shape: Vec<usize>) -> Result<Self, MaskError> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(MaskError::InvalidShape(
                "Shape cannot be empty or contain zeros".to_string(),
            ));
        }

        let mut indices = Vec::new();
        let total: usize = shape.iter().product();

        // Generate all possible indices
        for flat_idx in 0..total {
            let mut idx = Vec::with_capacity(shape.len());
            let mut remainder = flat_idx;

            for &dim_size in shape.iter().rev() {
                idx.push(remainder % dim_size);
                remainder /= dim_size;
            }
            idx.reverse();
            indices.push(idx);
        }

        Self::from_indices(indices, shape)
    }

    /// Check if an index is in the mask
    pub fn contains(&self, index: &[usize]) -> bool {
        self.indices.contains(index)
    }

    /// Get the number of active indices (mask nnz)
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get the shape of the masked tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements (including masked out)
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Compute the density (nnz / total)
    pub fn density(&self) -> f64 {
        self.nnz() as f64 / self.size() as f64
    }

    /// Iterator over active indices
    pub fn iter(&self) -> impl Iterator<Item = &Vec<usize>> {
        self.indices.iter()
    }

    /// Convert to sorted vector of indices
    pub fn to_sorted_indices(&self) -> Vec<Vec<usize>> {
        let mut indices: Vec<_> = self.indices.iter().cloned().collect();
        indices.sort();
        indices
    }

    /// Union of two masks (logical OR)
    pub fn union(&self, other: &Self) -> Result<Self, MaskError> {
        if self.shape != other.shape {
            return Err(MaskError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        let mut indices: HashSet<_> = self.indices.clone();
        indices.extend(other.indices.iter().cloned());

        Ok(Self {
            indices,
            shape: self.shape.clone(),
        })
    }

    /// Intersection of two masks (logical AND)
    pub fn intersection(&self, other: &Self) -> Result<Self, MaskError> {
        if self.shape != other.shape {
            return Err(MaskError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        let indices: HashSet<_> = self.indices.intersection(&other.indices).cloned().collect();

        Ok(Self {
            indices,
            shape: self.shape.clone(),
        })
    }

    /// Difference of two masks (self AND NOT other)
    pub fn difference(&self, other: &Self) -> Result<Self, MaskError> {
        if self.shape != other.shape {
            return Err(MaskError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        let indices: HashSet<_> = self.indices.difference(&other.indices).cloned().collect();

        Ok(Self {
            indices,
            shape: self.shape.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_creation() {
        let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
        let mask = Mask::from_indices(indices, vec![3, 4]).unwrap();

        assert_eq!(mask.nnz(), 3);
        assert_eq!(mask.shape(), &[3, 4]);
        assert_eq!(mask.rank(), 2);
        assert_eq!(mask.size(), 12);
    }

    #[test]
    fn test_mask_empty() {
        let mask = Mask::empty(vec![5, 5]).unwrap();
        assert_eq!(mask.nnz(), 0);
        assert_eq!(mask.size(), 25);
        assert_eq!(mask.density(), 0.0);
    }

    #[test]
    fn test_mask_full() {
        let mask = Mask::full(vec![2, 3]).unwrap();
        assert_eq!(mask.nnz(), 6);
        assert_eq!(mask.density(), 1.0);

        // Check all indices are present
        assert!(mask.contains(&[0, 0]));
        assert!(mask.contains(&[0, 1]));
        assert!(mask.contains(&[0, 2]));
        assert!(mask.contains(&[1, 0]));
        assert!(mask.contains(&[1, 1]));
        assert!(mask.contains(&[1, 2]));
    }

    #[test]
    fn test_mask_contains() {
        let indices = vec![vec![0, 1], vec![1, 2]];
        let mask = Mask::from_indices(indices, vec![3, 4]).unwrap();

        assert!(mask.contains(&[0, 1]));
        assert!(mask.contains(&[1, 2]));
        assert!(!mask.contains(&[0, 0]));
        assert!(!mask.contains(&[2, 3]));
    }

    #[test]
    fn test_mask_density() {
        let indices = vec![vec![0, 0], vec![1, 1]];
        let mask = Mask::from_indices(indices, vec![4, 4]).unwrap();

        assert_eq!(mask.density(), 2.0 / 16.0);
    }

    #[test]
    fn test_mask_union() {
        let mask1 = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![3, 3]).unwrap();
        let mask2 = Mask::from_indices(vec![vec![1, 1], vec![2, 2]], vec![3, 3]).unwrap();

        let union = mask1.union(&mask2).unwrap();
        assert_eq!(union.nnz(), 3);
        assert!(union.contains(&[0, 0]));
        assert!(union.contains(&[1, 1]));
        assert!(union.contains(&[2, 2]));
    }

    #[test]
    fn test_mask_intersection() {
        let mask1 = Mask::from_indices(vec![vec![0, 0], vec![1, 1]], vec![3, 3]).unwrap();
        let mask2 = Mask::from_indices(vec![vec![1, 1], vec![2, 2]], vec![3, 3]).unwrap();

        let intersection = mask1.intersection(&mask2).unwrap();
        assert_eq!(intersection.nnz(), 1);
        assert!(intersection.contains(&[1, 1]));
        assert!(!intersection.contains(&[0, 0]));
        assert!(!intersection.contains(&[2, 2]));
    }

    #[test]
    fn test_mask_difference() {
        let mask1 =
            Mask::from_indices(vec![vec![0, 0], vec![1, 1], vec![2, 2]], vec![3, 3]).unwrap();
        let mask2 = Mask::from_indices(vec![vec![1, 1]], vec![3, 3]).unwrap();

        let difference = mask1.difference(&mask2).unwrap();
        assert_eq!(difference.nnz(), 2);
        assert!(difference.contains(&[0, 0]));
        assert!(difference.contains(&[2, 2]));
        assert!(!difference.contains(&[1, 1]));
    }

    #[test]
    fn test_mask_out_of_bounds() {
        let indices = vec![vec![0, 5]]; // 5 is out of bounds for shape [3, 4]
        let result = Mask::from_indices(indices, vec![3, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_shape_mismatch_union() {
        let mask1 = Mask::empty(vec![3, 3]).unwrap();
        let mask2 = Mask::empty(vec![3, 4]).unwrap();

        let result = mask1.union(&mask2);
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_to_sorted_indices() {
        let indices = vec![vec![2, 1], vec![0, 0], vec![1, 2]];
        let mask = Mask::from_indices(indices, vec![3, 3]).unwrap();

        let sorted = mask.to_sorted_indices();
        assert_eq!(sorted.len(), 3);
        // Should be sorted
        assert!(sorted[0] <= sorted[1]);
        assert!(sorted[1] <= sorted[2]);
    }
}
