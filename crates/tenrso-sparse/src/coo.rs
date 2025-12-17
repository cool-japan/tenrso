//! COO (Coordinate) sparse tensor format
//!
//! The Coordinate format stores sparse tensors as a list of (coordinates, value) pairs.
//! This is the most flexible sparse format and serves as an intermediate for conversions.
//!
//! # Format
//!
//! For an N-dimensional sparse tensor:
//! - `indices`: `Vec<Vec<usize>>` - Each inner vec is one coordinate \[i₀, i₁, ..., iₙ₋₁\]
//! - `values`: `Vec<T>` - The non-zero values
//! - `shape`: `Vec<usize>` - The shape of the tensor
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::coo::CooTensor;
//!
//! // Create a 3x4 sparse matrix with 3 non-zero elements
//! let indices = vec![
//!     vec![0, 1],  // (0,1) = 2.5
//!     vec![1, 2],  // (1,2) = 3.0
//!     vec![2, 0],  // (2,0) = 1.5
//! ];
//! let values = vec![2.5, 3.0, 1.5];
//! let shape = vec![3, 4];
//!
//! let coo = CooTensor::new(indices, values, shape).unwrap();
//! assert_eq!(coo.nnz(), 3);
//! assert_eq!(coo.shape(), &[3, 4]);
//! ```
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::numeric::Float;
use tenrso_core::DenseND;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CooError {
    #[error(
        "Shape mismatch: indices have {indices_len} elements but shape has {shape_len} dimensions"
    )]
    ShapeMismatch {
        indices_len: usize,
        shape_len: usize,
    },

    #[error("Length mismatch: {indices} indices but {values} values")]
    LengthMismatch { indices: usize, values: usize },

    #[error("Index out of bounds: index {index:?} exceeds shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Empty tensor")]
    EmptyTensor,
}

/// COO (Coordinate) sparse tensor
///
/// Stores sparse tensors as (coordinate, value) pairs.
/// Flexible and easy to construct, but not optimized for operations.
#[derive(Debug, Clone)]
pub struct CooTensor<T> {
    /// Coordinates of non-zero elements
    /// Each inner vec is [i₀, i₁, ..., iₙ₋₁]
    indices: Vec<Vec<usize>>,

    /// Values at the corresponding coordinates
    values: Vec<T>,

    /// Shape of the tensor
    shape: Vec<usize>,
}

impl<T: Clone> CooTensor<T> {
    /// Create a new COO sparse tensor
    ///
    /// # Arguments
    ///
    /// * `indices` - Coordinates of non-zero elements
    /// * `values` - Values at those coordinates
    /// * `shape` - Shape of the tensor
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Indices and values have different lengths
    /// - Indices dimensionality doesn't match shape
    /// - Any index is out of bounds
    /// - Shape contains zeros
    pub fn new(
        indices: Vec<Vec<usize>>,
        values: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self, CooError> {
        // Validate lengths
        if indices.len() != values.len() {
            return Err(CooError::LengthMismatch {
                indices: indices.len(),
                values: values.len(),
            });
        }

        // Validate shape
        if shape.is_empty() {
            return Err(CooError::InvalidShape("Shape cannot be empty".to_string()));
        }
        if shape.contains(&0) {
            return Err(CooError::InvalidShape(
                "Shape cannot contain zeros".to_string(),
            ));
        }

        // Validate indices
        for idx in &indices {
            if idx.len() != shape.len() {
                return Err(CooError::ShapeMismatch {
                    indices_len: idx.len(),
                    shape_len: shape.len(),
                });
            }

            // Check bounds
            for (&coord, &size) in idx.iter().zip(&shape) {
                if coord >= size {
                    return Err(CooError::IndexOutOfBounds {
                        index: idx.clone(),
                        shape: shape.clone(),
                    });
                }
            }
        }

        Ok(Self {
            indices,
            values,
            shape,
        })
    }

    /// Create an empty COO tensor with given shape
    pub fn zeros(shape: Vec<usize>) -> Result<Self, CooError> {
        if shape.is_empty() {
            return Err(CooError::InvalidShape("Shape cannot be empty".to_string()));
        }
        if shape.contains(&0) {
            return Err(CooError::InvalidShape(
                "Shape cannot contain zeros".to_string(),
            ));
        }

        Ok(Self {
            indices: Vec::new(),
            values: Vec::new(),
            shape,
        })
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get indices
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    /// Get values
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Get mutable reference to indices
    pub fn indices_mut(&mut self) -> &mut [Vec<usize>] {
        &mut self.indices
    }

    /// Get mutable reference to values
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Compute density (nnz / total_elements)
    pub fn density(&self) -> f64 {
        let total: usize = self.shape.iter().product();
        self.nnz() as f64 / total as f64
    }

    /// Add a non-zero element
    ///
    /// Note: Does not check for duplicates. Use `deduplicate()` after construction.
    pub fn push(&mut self, index: Vec<usize>, value: T) -> Result<(), CooError> {
        // Validate index
        if index.len() != self.shape.len() {
            return Err(CooError::ShapeMismatch {
                indices_len: index.len(),
                shape_len: self.shape.len(),
            });
        }

        for (&coord, &size) in index.iter().zip(&self.shape) {
            if coord >= size {
                return Err(CooError::IndexOutOfBounds {
                    index,
                    shape: self.shape.clone(),
                });
            }
        }

        self.indices.push(index);
        self.values.push(value);
        Ok(())
    }

    /// Sort indices in row-major (C-contiguous) order
    ///
    /// This is useful for converting to other formats like CSR.
    pub fn sort(&mut self)
    where
        T: Clone,
    {
        // Create index permutation
        let mut perm: Vec<usize> = (0..self.nnz()).collect();

        // Sort by indices in row-major order
        perm.sort_by(|&i, &j| self.indices[i].cmp(&self.indices[j]));

        // Apply permutation
        let old_indices = self.indices.clone();
        let old_values = self.values.clone();

        for (new_idx, &old_idx) in perm.iter().enumerate() {
            self.indices[new_idx] = old_indices[old_idx].clone();
            self.values[new_idx] = old_values[old_idx].clone();
        }
    }
}

impl<T: Float> CooTensor<T> {
    /// Convert to dense tensor
    ///
    /// # Complexity
    ///
    /// Time: O(nnz)
    /// Space: O(∏ᵢ shape\[i\])
    pub fn to_dense(&self) -> Result<DenseND<T>> {
        let total_size: usize = self.shape.iter().product();
        let mut data = vec![T::zero(); total_size];

        // Fill in non-zero values
        for (idx, &value) in self.indices.iter().zip(&self.values) {
            // Compute linear index from multi-index (row-major order)
            let mut linear_idx = 0;
            let mut stride = 1;
            for (dim, &coord) in idx.iter().enumerate().rev() {
                linear_idx += coord * stride;
                stride *= self.shape[dim];
            }
            data[linear_idx] = value;
        }

        DenseND::from_vec(data, &self.shape)
    }

    /// Create COO tensor from dense tensor
    ///
    /// Only stores elements where |value| > threshold.
    pub fn from_dense(dense: &DenseND<T>, threshold: T) -> Self {
        let shape = dense.shape().to_vec();
        let view = dense.view();

        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Iterate through all elements
        let total_size: usize = shape.iter().product();
        for flat_idx in 0..total_size {
            // Convert linear index to multi-index
            let mut multi_idx = vec![0; shape.len()];
            let mut remaining = flat_idx;
            for (dim, &size) in shape.iter().enumerate().rev() {
                multi_idx[dim] = remaining % size;
                remaining /= size;
            }

            // Get value
            let value = view[&multi_idx[..]];
            if value.abs() > threshold {
                indices.push(multi_idx);
                values.push(value);
            }
        }

        // Safe to unwrap: we control the construction
        Self::new(indices, values, shape).unwrap()
    }

    /// Deduplicate entries by summing values at the same coordinate
    pub fn deduplicate(&mut self) {
        if self.nnz() <= 1 {
            return;
        }

        // First sort
        self.sort();

        // Then deduplicate consecutive equal indices
        let mut write_idx = 0;
        for read_idx in 1..self.nnz() {
            if self.indices[write_idx] == self.indices[read_idx] {
                // Same coordinate, sum values
                self.values[write_idx] = self.values[write_idx] + self.values[read_idx];
            } else {
                // Different coordinate, move to next write position
                write_idx += 1;
                if write_idx != read_idx {
                    self.indices[write_idx] = self.indices[read_idx].clone();
                    self.values[write_idx] = self.values[read_idx];
                }
            }
        }

        // Truncate to deduplicated size
        self.indices.truncate(write_idx + 1);
        self.values.truncate(write_idx + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_creation() {
        let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
        let values = vec![2.5, 3.0, 1.5];
        let shape = vec![3, 4];

        let coo = CooTensor::new(indices, values, shape).unwrap();
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape(), &[3, 4]);
        assert_eq!(coo.rank(), 2);
    }

    #[test]
    fn test_coo_zeros() {
        let coo = CooTensor::<f64>::zeros(vec![5, 5]).unwrap();
        assert_eq!(coo.nnz(), 0);
        assert_eq!(coo.shape(), &[5, 5]);
    }

    #[test]
    fn test_coo_density() {
        let indices = vec![vec![0, 0], vec![1, 1]];
        let values = vec![1.0, 2.0];
        let shape = vec![10, 10];

        let coo = CooTensor::new(indices, values, shape).unwrap();
        assert_eq!(coo.density(), 0.02); // 2/100
    }

    #[test]
    fn test_coo_push() {
        let mut coo = CooTensor::<f64>::zeros(vec![3, 3]).unwrap();
        coo.push(vec![0, 0], 1.0).unwrap();
        coo.push(vec![1, 1], 2.0).unwrap();

        assert_eq!(coo.nnz(), 2);
    }

    #[test]
    fn test_coo_to_dense() {
        let indices = vec![vec![0, 1], vec![1, 0], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3];

        let coo = CooTensor::new(indices, values, shape).unwrap();
        let dense = coo.to_dense().unwrap();

        assert_eq!(dense.shape(), &[3, 3]);
        let view = dense.view();
        assert_eq!(view[[0, 1]], 1.0);
        assert_eq!(view[[1, 0]], 2.0);
        assert_eq!(view[[2, 2]], 3.0);
        assert_eq!(view[[0, 0]], 0.0);
    }

    #[test]
    fn test_coo_from_dense() {
        let dense =
            DenseND::<f64>::from_vec(vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
                .unwrap();

        let coo = CooTensor::from_dense(&dense, 1e-10);
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_coo_deduplicate() {
        let indices = vec![
            vec![0, 0],
            vec![0, 0], // duplicate
            vec![1, 1],
            vec![1, 1], // duplicate
            vec![2, 2],
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let mut coo = CooTensor::new(indices, values, shape).unwrap();
        coo.deduplicate();

        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.values(), &[3.0, 7.0, 5.0]); // Summed values
    }

    #[test]
    fn test_coo_sort() {
        let indices = vec![vec![2, 0], vec![0, 1], vec![1, 0]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3];

        let mut coo = CooTensor::new(indices, values, shape).unwrap();
        coo.sort();

        // Should be sorted: [0,1], [1,0], [2,0]
        assert_eq!(coo.indices()[0], vec![0, 1]);
        assert_eq!(coo.indices()[1], vec![1, 0]);
        assert_eq!(coo.indices()[2], vec![2, 0]);
    }

    #[test]
    fn test_coo_3d() {
        let indices = vec![vec![0, 1, 2], vec![1, 0, 1], vec![2, 2, 0]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3, 3];

        let coo = CooTensor::new(indices, values, shape).unwrap();
        assert_eq!(coo.rank(), 3);
        assert_eq!(coo.nnz(), 3);
    }
}
