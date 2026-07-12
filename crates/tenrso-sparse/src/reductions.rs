//! Reduction operations for sparse tensors
//!
//! This module provides reduction operations (sum, max, min, mean, etc.) for sparse tensors.
//! Reductions can be performed globally or along specific axes.
//!
//! # Complexity
//!
//! - Global reductions: O(nnz) where nnz is the number of non-zero elements
//! - Axis-wise reductions: O(nnz) with result size dependent on the reduced tensor dimensions
//!
//! # Examples
//!
//! ```
//! use tenrso_sparse::coo::CooTensor;
//! use tenrso_sparse::reductions::*;
//!
//! // Create a sparse 3x4 matrix
//! let indices = vec![
//!     vec![0, 1],  // (0,1) = 2.0
//!     vec![1, 2],  // (1,2) = 3.0
//!     vec![2, 0],  // (2,0) = 1.0
//! ];
//! let values = vec![2.0, 3.0, 1.0];
//! let shape = vec![3, 4];
//! let coo = CooTensor::new(indices, values, shape).unwrap();
//!
//! // Global sum
//! let total = sum(&coo);
//! assert_eq!(total, 6.0);
//!
//! // Sum along axis 0 (row reduction) -> 4-element vector
//! let row_sums = sum_axis(&coo, 0).unwrap();
//! assert_eq!(row_sums.nnz(), 3);
//! ```
//!
//! # SciRS2 Integration
//!
//! All numeric operations use `scirs2_core::numeric::Float`.
//! Direct use of standard library traits is avoided per SCIRS2_INTEGRATION_POLICY.md

use crate::coo::{CooError, CooTensor};
use anyhow::Result;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReductionError {
    #[error("Invalid axis: {axis} for tensor with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("Empty tensor: cannot compute {operation}")]
    EmptyTensor { operation: String },

    #[error("COO error: {0}")]
    CooError(#[from] CooError),
}

/// Global sum reduction
///
/// Computes the sum of all elements in the sparse tensor, including implicit zeros.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::sum;
///
/// let indices = vec![vec![0], vec![2], vec![4]];
/// let values = vec![1.0, 2.0, 3.0];
/// let coo = CooTensor::new(indices, values, vec![5]).unwrap();
///
/// assert_eq!(sum(&coo), 6.0);
/// ```
pub fn sum<T: Float>(tensor: &CooTensor<T>) -> T {
    tensor.values().iter().fold(T::zero(), |acc, &v| acc + v)
}

/// Global product reduction
///
/// Computes the product of all elements in the sparse tensor.
/// Returns zero if there are any implicit zeros (sparsity > 0).
///
/// # Complexity
///
/// O(1) if tensor has any zeros (returns 0 immediately)
/// O(nnz) otherwise
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::product;
///
/// let indices = vec![vec![0], vec![1], vec![2]];
/// let values = vec![2.0, 3.0, 4.0];
/// let coo = CooTensor::new(indices, values, vec![3]).unwrap();
///
/// assert_eq!(product(&coo), 24.0);
///
/// // Sparse tensor with implicit zeros
/// let indices2 = vec![vec![0], vec![2]];
/// let values2 = vec![2.0, 3.0];
/// let sparse_coo = CooTensor::new(indices2, values2, vec![5]).unwrap();
/// assert_eq!(product(&sparse_coo), 0.0); // Has implicit zeros
/// ```
pub fn product<T: Float>(tensor: &CooTensor<T>) -> T {
    let total_elements: usize = tensor.shape().iter().product();

    // If we have any implicit zeros, product is zero
    if tensor.nnz() < total_elements {
        return T::zero();
    }

    // All elements are non-zero, compute product
    tensor.values().iter().fold(T::one(), |acc, &v| acc * v)
}

/// Global maximum reduction
///
/// Computes the maximum element in the sparse tensor, including implicit zeros.
///
/// # Complexity
///
/// O(nnz)
///
/// # Errors
///
/// Returns error if tensor is empty (has no shape)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::max;
///
/// let indices = vec![vec![0], vec![2]];
/// let values = vec![5.0, -3.0];
/// let coo = CooTensor::new(indices, values, vec![5]).unwrap();
///
/// assert_eq!(max(&coo).unwrap(), 5.0);
///
/// // All negative values with implicit zeros
/// let neg_coo = CooTensor::new(
///     vec![vec![0], vec![1]],
///     vec![-1.0, -2.0],
///     vec![5]
/// ).unwrap();
/// assert_eq!(max(&neg_coo).unwrap(), 0.0); // Implicit zero is max
/// ```
pub fn max<T: Float>(tensor: &CooTensor<T>) -> Result<T> {
    if tensor.shape().is_empty() {
        return Err(ReductionError::EmptyTensor {
            operation: "max".to_string(),
        }
        .into());
    }

    let total_elements: usize = tensor.shape().iter().product();

    // If tensor is completely empty (all implicit zeros)
    if tensor.nnz() == 0 {
        return Ok(T::zero());
    }

    // Find max among non-zero values
    let max_nonzero =
        tensor
            .values()
            .iter()
            .fold(tensor.values()[0], |acc, &v| if v > acc { v } else { acc });

    // If we have implicit zeros, compare with zero
    if tensor.nnz() < total_elements {
        Ok(if max_nonzero > T::zero() {
            max_nonzero
        } else {
            T::zero()
        })
    } else {
        Ok(max_nonzero)
    }
}

/// Global minimum reduction
///
/// Computes the minimum element in the sparse tensor, including implicit zeros.
///
/// # Complexity
///
/// O(nnz)
///
/// # Errors
///
/// Returns error if tensor is empty (has no shape)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::min;
///
/// let indices = vec![vec![0], vec![2]];
/// let values = vec![5.0, 3.0];
/// let coo = CooTensor::new(indices, values, vec![5]).unwrap();
///
/// assert_eq!(min(&coo).unwrap(), 0.0); // Implicit zero is min
///
/// // All negative values
/// let neg_coo = CooTensor::new(
///     vec![vec![0], vec![1]],
///     vec![-1.0, -5.0],
///     vec![2]
/// ).unwrap();
/// assert_eq!(min(&neg_coo).unwrap(), -5.0);
/// ```
pub fn min<T: Float>(tensor: &CooTensor<T>) -> Result<T> {
    if tensor.shape().is_empty() {
        return Err(ReductionError::EmptyTensor {
            operation: "min".to_string(),
        }
        .into());
    }

    let total_elements: usize = tensor.shape().iter().product();

    // If tensor is completely empty (all implicit zeros)
    if tensor.nnz() == 0 {
        return Ok(T::zero());
    }

    // Find min among non-zero values
    let min_nonzero =
        tensor
            .values()
            .iter()
            .fold(tensor.values()[0], |acc, &v| if v < acc { v } else { acc });

    // If we have implicit zeros, compare with zero
    if tensor.nnz() < total_elements {
        Ok(if min_nonzero < T::zero() {
            min_nonzero
        } else {
            T::zero()
        })
    } else {
        Ok(min_nonzero)
    }
}

/// Global mean reduction
///
/// Computes the mean of all elements in the sparse tensor, including implicit zeros.
///
/// # Complexity
///
/// O(nnz)
///
/// # Errors
///
/// Returns error if tensor is empty (has no shape)
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::mean;
///
/// let indices = vec![vec![0], vec![2]];
/// let values = vec![4.0, 6.0];
/// let coo = CooTensor::new(indices, values, vec![5]).unwrap();
///
/// // (4 + 6 + 0 + 0 + 0) / 5 = 2.0
/// assert_eq!(mean(&coo).unwrap(), 2.0);
/// ```
pub fn mean<T: Float>(tensor: &CooTensor<T>) -> Result<T> {
    if tensor.shape().is_empty() {
        return Err(ReductionError::EmptyTensor {
            operation: "mean".to_string(),
        }
        .into());
    }

    let total_elements: usize = tensor.shape().iter().product();
    let total_sum = sum(tensor);

    // Convert total_elements to T for division
    let n = T::from(total_elements as f64).unwrap_or_else(|| T::one());

    Ok(total_sum / n)
}

/// Sum reduction along a specific axis
///
/// Reduces the tensor along the specified axis by summing elements.
/// The result has one fewer dimension than the input.
///
/// # Complexity
///
/// O(nnz) where nnz is the number of non-zero elements
///
/// # Arguments
///
/// * `tensor` - The sparse tensor to reduce
/// * `axis` - The axis to reduce along (0-indexed)
///
/// # Errors
///
/// Returns error if axis is out of bounds
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::sum_axis;
///
/// // Create a 3x4 matrix
/// let indices = vec![
///     vec![0, 1],  // (0,1) = 2.0
///     vec![1, 2],  // (1,2) = 3.0
///     vec![2, 0],  // (2,0) = 1.0
/// ];
/// let values = vec![2.0, 3.0, 1.0];
/// let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();
///
/// // Sum along axis 0 (collapse rows) -> [1, 0, 3, 0]
/// let col_sums = sum_axis(&coo, 0).unwrap();
/// assert_eq!(col_sums.shape(), &[4]);
/// ```
pub fn sum_axis<T: Float + Clone + std::ops::AddAssign>(
    tensor: &CooTensor<T>,
    axis: usize,
) -> Result<CooTensor<T>> {
    if axis >= tensor.shape().len() {
        return Err(ReductionError::InvalidAxis {
            axis,
            ndim: tensor.shape().len(),
        }
        .into());
    }

    // Build new shape (remove the reduced axis)
    let mut new_shape = tensor.shape().to_vec();
    new_shape.remove(axis);

    // Handle edge case: reducing last dimension leaves scalar
    if new_shape.is_empty() {
        // Result is a 0-D tensor (scalar), but COO can't represent this
        // Return a 1-D tensor with 1 element
        let total = sum(tensor);
        return Ok(CooTensor::new(vec![vec![0]], vec![total], vec![1])?);
    }

    // Accumulate sums by building indices without the reduced axis
    let mut sum_map: HashMap<Vec<usize>, T> = HashMap::new();

    for (idx, &val) in tensor.indices().iter().zip(tensor.values().iter()) {
        // Project index by removing the axis dimension
        let mut proj_idx = idx.clone();
        proj_idx.remove(axis);

        *sum_map.entry(proj_idx).or_insert_with(T::zero) += val;
    }

    // Convert HashMap to COO format
    let mut result_indices = Vec::with_capacity(sum_map.len());
    let mut result_values = Vec::with_capacity(sum_map.len());

    for (idx, val) in sum_map {
        // Only keep non-zero values
        if val.abs() > T::epsilon() {
            result_indices.push(idx);
            result_values.push(val);
        }
    }

    CooTensor::new(result_indices, result_values, new_shape).map_err(|e| e.into())
}

/// Maximum reduction along a specific axis
///
/// Reduces the tensor along the specified axis by taking the maximum element.
/// The result has one fewer dimension than the input.
///
/// # Complexity
///
/// O(nnz + result_size) where result_size accounts for implicit zeros
///
/// # Arguments
///
/// * `tensor` - The sparse tensor to reduce
/// * `axis` - The axis to reduce along (0-indexed)
///
/// # Errors
///
/// Returns error if axis is out of bounds
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::max_axis;
///
/// // Create a 3x4 matrix
/// let indices = vec![
///     vec![0, 1],  // (0,1) = 5.0
///     vec![1, 2],  // (1,2) = 3.0
///     vec![2, 0],  // (2,0) = 1.0
/// ];
/// let values = vec![5.0, 3.0, 1.0];
/// let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();
///
/// // Max along axis 0 (max of each column)
/// let col_max = max_axis(&coo, 0).unwrap();
/// assert_eq!(col_max.shape(), &[4]);
/// // Column 0: max(0, 0, 1) = 1
/// // Column 1: max(5, 0, 0) = 5
/// // Column 2: max(0, 3, 0) = 3
/// // Column 3: max(0, 0, 0) = 0
/// ```
pub fn max_axis<T: Float + Clone>(tensor: &CooTensor<T>, axis: usize) -> Result<CooTensor<T>> {
    if axis >= tensor.shape().len() {
        return Err(ReductionError::InvalidAxis {
            axis,
            ndim: tensor.shape().len(),
        }
        .into());
    }

    // Build new shape (remove the reduced axis)
    let mut new_shape = tensor.shape().to_vec();
    let axis_size = new_shape.remove(axis);

    // Handle edge case: reducing last dimension leaves scalar
    if new_shape.is_empty() {
        let max_val = max(tensor)?;
        return Ok(CooTensor::new(vec![vec![0]], vec![max_val], vec![1])?);
    }

    // For max reductions we must be careful about implicit zeros. The implicit
    // zero of a missing entry only participates in the maximum when the slice is
    // NOT fully populated along the reduction axis. A slice that is fully dense
    // with all-negative values has its true maximum equal to the largest (least
    // negative) observed value, never 0. Track the observed count per slice so we
    // can distinguish "has a structural zero" from "fully populated".
    let mut max_map: HashMap<Vec<usize>, T> = HashMap::new();
    let mut count_map: HashMap<Vec<usize>, usize> = HashMap::new();

    for (idx, &val) in tensor.indices().iter().zip(tensor.values().iter()) {
        // Project index by removing the axis dimension
        let mut proj_idx = idx.clone();
        proj_idx.remove(axis);

        *count_map.entry(proj_idx.clone()).or_insert(0) += 1;

        max_map
            .entry(proj_idx.clone())
            .and_modify(|current| {
                if val > *current {
                    *current = val;
                }
            })
            .or_insert(val);
    }

    // Build result - only include non-zero values.
    // Slices that are entirely implicit (no observed entries) never appear in
    // `max_map`; their maximum is the implicit 0, which is omitted from the
    // sparse result by construction.
    let mut result_indices = Vec::new();
    let mut result_values = Vec::new();

    for (idx, val) in max_map {
        let count = count_map
            .get(&idx)
            .expect("count_map has the same keys as max_map by construction");

        // If we haven't observed every position along the reduced axis, the slice
        // contains at least one structural (implicit) zero that must be folded in.
        let has_implicit_zeros = *count < axis_size;

        let final_max = if has_implicit_zeros {
            // The implicit zero participates: max(max_observed, 0).
            if val > T::zero() {
                val
            } else {
                T::zero()
            }
        } else {
            // Slice is fully populated; the true max is the max of observed values.
            val
        };

        // Only include if non-zero
        if final_max.abs() > T::epsilon() {
            result_indices.push(idx);
            result_values.push(final_max);
        }
    }

    CooTensor::new(result_indices, result_values, new_shape).map_err(|e| e.into())
}

/// Minimum reduction along a specific axis
///
/// Reduces the tensor along the specified axis by taking the minimum element.
/// The result has one fewer dimension than the input.
///
/// # Complexity
///
/// O(nnz + result_size)
///
/// # Arguments
///
/// * `tensor` - The sparse tensor to reduce
/// * `axis` - The axis to reduce along (0-indexed)
///
/// # Errors
///
/// Returns error if axis is out of bounds
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::min_axis;
///
/// // Create a 3x4 matrix with positive values
/// let indices = vec![
///     vec![0, 1],  // (0,1) = 5.0
///     vec![1, 2],  // (1,2) = 3.0
///     vec![2, 0],  // (2,0) = 1.0
/// ];
/// let values = vec![5.0, 3.0, 1.0];
/// let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();
///
/// // Min along axis 0 (min of each column)
/// let col_min = min_axis(&coo, 0).unwrap();
/// assert_eq!(col_min.shape(), &[4]);
/// // All columns have at least one implicit zero, so all mins are 0
/// assert_eq!(col_min.nnz(), 0); // All zeros (sparse result)
/// ```
pub fn min_axis<T: Float + Clone>(tensor: &CooTensor<T>, axis: usize) -> Result<CooTensor<T>> {
    if axis >= tensor.shape().len() {
        return Err(ReductionError::InvalidAxis {
            axis,
            ndim: tensor.shape().len(),
        }
        .into());
    }

    // Build new shape (remove the reduced axis)
    let mut new_shape = tensor.shape().to_vec();
    let axis_size = new_shape.remove(axis);

    // Handle edge case: reducing last dimension leaves scalar
    if new_shape.is_empty() {
        let min_val = min(tensor)?;
        return Ok(CooTensor::new(vec![vec![0]], vec![min_val], vec![1])?);
    }

    // Track minimum values and whether we've seen any values
    let mut min_map: HashMap<Vec<usize>, T> = HashMap::new();
    let mut count_map: HashMap<Vec<usize>, usize> = HashMap::new();

    for (idx, &val) in tensor.indices().iter().zip(tensor.values().iter()) {
        // Project index by removing the axis dimension
        let mut proj_idx = idx.clone();
        proj_idx.remove(axis);

        *count_map.entry(proj_idx.clone()).or_insert(0) += 1;

        min_map
            .entry(proj_idx.clone())
            .and_modify(|current| {
                if val < *current {
                    *current = val;
                }
            })
            .or_insert(val);
    }

    // Build result
    let mut result_indices = Vec::new();
    let mut result_values = Vec::new();

    for (idx, val) in min_map {
        let count = count_map
            .get(&idx)
            .expect("count_map has the same keys as min_map by construction");

        // If we haven't seen all positions along this axis, there are implicit zeros
        let has_implicit_zeros = *count < axis_size;

        let final_min = if has_implicit_zeros {
            // Compare with zero
            if val < T::zero() {
                val
            } else {
                T::zero()
            }
        } else {
            val
        };

        // Only include if non-zero
        if final_min.abs() > T::epsilon() {
            result_indices.push(idx);
            result_values.push(final_min);
        }
    }

    CooTensor::new(result_indices, result_values, new_shape).map_err(|e| e.into())
}

/// Mean reduction along a specific axis
///
/// Reduces the tensor along the specified axis by computing the mean.
/// The result has one fewer dimension than the input.
///
/// # Complexity
///
/// O(nnz)
///
/// # Arguments
///
/// * `tensor` - The sparse tensor to reduce
/// * `axis` - The axis to reduce along (0-indexed)
///
/// # Errors
///
/// Returns error if axis is out of bounds
///
/// # Examples
///
/// ```
/// use tenrso_sparse::coo::CooTensor;
/// use tenrso_sparse::reductions::mean_axis;
///
/// // Create a 3x4 matrix
/// let indices = vec![
///     vec![0, 1],  // (0,1) = 6.0
///     vec![1, 2],  // (1,2) = 9.0
///     vec![2, 0],  // (2,0) = 3.0
/// ];
/// let values = vec![6.0, 9.0, 3.0];
/// let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();
///
/// // Mean along axis 0 (average of each column over 3 rows)
/// let col_mean = mean_axis(&coo, 0).unwrap();
/// assert_eq!(col_mean.shape(), &[4]);
/// // Column 0: (0+0+3)/3 = 1.0
/// // Column 1: (6+0+0)/3 = 2.0
/// // Column 2: (0+9+0)/3 = 3.0
/// // Column 3: (0+0+0)/3 = 0.0
/// ```
pub fn mean_axis<T: Float + Clone + std::ops::AddAssign>(
    tensor: &CooTensor<T>,
    axis: usize,
) -> Result<CooTensor<T>> {
    if axis >= tensor.shape().len() {
        return Err(ReductionError::InvalidAxis {
            axis,
            ndim: tensor.shape().len(),
        }
        .into());
    }

    // Get the size of the axis we're reducing over
    let axis_size = tensor.shape()[axis];
    let divisor = T::from(axis_size as f64).unwrap_or_else(|| T::one());

    // First compute sum along axis
    let sum_result = sum_axis(tensor, axis)?;

    // Then divide each element by axis_size
    let mean_values: Vec<T> = sum_result.values().iter().map(|&v| v / divisor).collect();

    CooTensor::new(
        sum_result.indices().to_vec(),
        mean_values,
        sum_result.shape().to_vec(),
    )
    .map_err(|e| e.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_sum() {
        let indices = vec![vec![0], vec![2], vec![4]];
        let values = vec![1.0, 2.0, 3.0];
        let coo = CooTensor::new(indices, values, vec![5]).unwrap();

        assert_eq!(sum(&coo), 6.0);
    }

    #[test]
    fn test_global_sum_empty() {
        let coo: CooTensor<f64> = CooTensor::zeros(vec![3, 4]).unwrap();
        assert_eq!(sum(&coo), 0.0);
    }

    #[test]
    fn test_global_product_dense() {
        let indices = vec![vec![0], vec![1], vec![2]];
        let values = vec![2.0, 3.0, 4.0];
        let coo = CooTensor::new(indices, values, vec![3]).unwrap();

        assert_eq!(product(&coo), 24.0);
    }

    #[test]
    fn test_global_product_sparse() {
        let indices = vec![vec![0], vec![2]];
        let values = vec![2.0, 3.0];
        let coo = CooTensor::new(indices, values, vec![5]).unwrap();

        // Has implicit zeros, product is 0
        assert_eq!(product(&coo), 0.0);
    }

    #[test]
    fn test_global_max() {
        let indices = vec![vec![0], vec![2]];
        let values = vec![5.0, -3.0];
        let coo = CooTensor::new(indices, values, vec![5]).unwrap();

        assert_eq!(max(&coo).unwrap(), 5.0);
    }

    #[test]
    fn test_global_max_with_implicit_zeros() {
        let indices = vec![vec![0], vec![1]];
        let values = vec![-1.0, -2.0];
        let coo = CooTensor::new(indices, values, vec![5]).unwrap();

        // Implicit zeros are max
        assert_eq!(max(&coo).unwrap(), 0.0);
    }

    #[test]
    fn test_global_min() {
        let indices = vec![vec![0], vec![2]];
        let values = vec![5.0, 3.0];
        let coo = CooTensor::new(indices, values, vec![5]).unwrap();

        // Implicit zeros are min
        assert_eq!(min(&coo).unwrap(), 0.0);
    }

    #[test]
    fn test_global_min_negative() {
        let indices = vec![vec![0], vec![1]];
        let values = vec![-1.0, -5.0];
        let coo = CooTensor::new(indices, values, vec![2]).unwrap();

        assert_eq!(min(&coo).unwrap(), -5.0);
    }

    #[test]
    fn test_global_mean() {
        let indices = vec![vec![0], vec![2]];
        let values = vec![4.0, 6.0];
        let coo = CooTensor::new(indices, values, vec![5]).unwrap();

        // (4 + 6 + 0 + 0 + 0) / 5 = 2.0
        assert_eq!(mean(&coo).unwrap(), 2.0);
    }

    #[test]
    fn test_sum_axis_matrix_axis0() {
        // 3x4 matrix, sum along rows (collapse to 4-element vector)
        let indices = vec![
            vec![0, 1], // (0,1) = 2.0
            vec![1, 2], // (1,2) = 3.0
            vec![2, 0], // (2,0) = 1.0
        ];
        let values = vec![2.0, 3.0, 1.0];
        let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();

        let result = sum_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.nnz(), 3);

        // Verify sums: [1, 2, 3, 0]
        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        assert_eq!(data[[0]], 1.0);
        assert_eq!(data[[1]], 2.0);
        assert_eq!(data[[2]], 3.0);
        assert_eq!(data[[3]], 0.0);
    }

    #[test]
    fn test_sum_axis_matrix_axis1() {
        // 3x4 matrix, sum along columns (collapse to 3-element vector)
        let indices = vec![
            vec![0, 1], // (0,1) = 2.0
            vec![1, 2], // (1,2) = 3.0
            vec![2, 0], // (2,0) = 1.0
        ];
        let values = vec![2.0, 3.0, 1.0];
        let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();

        let result = sum_axis(&coo, 1).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.nnz(), 3);

        // Row sums: [2, 3, 1]
        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        assert_eq!(data[[0]], 2.0);
        assert_eq!(data[[1]], 3.0);
        assert_eq!(data[[2]], 1.0);
    }

    #[test]
    fn test_sum_axis_3d() {
        // 2x3x4 tensor
        let indices = vec![
            vec![0, 0, 0], // (0,0,0) = 1.0
            vec![0, 1, 2], // (0,1,2) = 2.0
            vec![1, 2, 1], // (1,2,1) = 3.0
        ];
        let values = vec![1.0, 2.0, 3.0];
        let coo = CooTensor::new(indices, values, vec![2, 3, 4]).unwrap();

        // Sum along axis 1 -> shape [2, 4]
        let result = sum_axis(&coo, 1).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(result.nnz(), 3);
    }

    #[test]
    fn test_max_axis() {
        let indices = vec![
            vec![0, 1], // (0,1) = 5.0
            vec![1, 2], // (1,2) = 3.0
            vec![2, 0], // (2,0) = 1.0
        ];
        let values = vec![5.0, 3.0, 1.0];
        let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();

        let result = max_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[4]);

        // Column maxes: [1, 5, 3, 0]
        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        assert_eq!(data[[0]], 1.0);
        assert_eq!(data[[1]], 5.0);
        assert_eq!(data[[2]], 3.0);
        assert_eq!(data[[3]], 0.0);
    }

    #[test]
    fn test_min_axis_with_negatives() {
        let indices = vec![
            vec![0, 1], // (0,1) = -5.0
            vec![1, 2], // (1,2) = -3.0
            vec![2, 0], // (2,0) = -1.0
        ];
        let values = vec![-5.0, -3.0, -1.0];
        let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();

        let result = min_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[4]);

        // Column mins (with implicit zeros): [-1, -5, -3, 0]
        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        assert_eq!(data[[0]], -1.0);
        assert_eq!(data[[1]], -5.0);
        assert_eq!(data[[2]], -3.0);
        assert_eq!(data[[3]], 0.0);
    }

    #[test]
    fn test_mean_axis() {
        let indices = vec![
            vec![0, 1], // (0,1) = 6.0
            vec![1, 2], // (1,2) = 9.0
            vec![2, 0], // (2,0) = 3.0
        ];
        let values = vec![6.0, 9.0, 3.0];
        let coo = CooTensor::new(indices, values, vec![3, 4]).unwrap();

        let result = mean_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[4]);

        // Column means: [1, 2, 3, 0]
        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        assert!((data[[0]] - 1.0).abs() < 1e-10);
        assert!((data[[1]] - 2.0).abs() < 1e-10);
        assert!((data[[2]] - 3.0).abs() < 1e-10);
        assert!((data[[3]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduction_invalid_axis() {
        let coo: CooTensor<f64> = CooTensor::zeros(vec![3, 4]).unwrap();

        assert!(sum_axis(&coo, 2).is_err());
        assert!(max_axis(&coo, 2).is_err());
        assert!(min_axis(&coo, 2).is_err());
        assert!(mean_axis(&coo, 2).is_err());
    }

    // ------------------------------------------------------------------
    // Regression tests for the "implicit zero" clamp bug in max_axis.
    //
    // The old code unconditionally clamped every reduced slice's maximum
    // against 0. For a slice that is FULLY populated along the reduction
    // axis with all-negative values, the true maximum is the largest
    // (least negative) observed value, NOT 0. These tests fail against the
    // old clamp and pass with the count-aware fix.
    // ------------------------------------------------------------------

    /// Fully-dense, all-negative slices: max must be the least-negative
    /// observed value, never the implicit 0 (there is no implicit zero).
    #[test]
    fn test_max_axis_fully_dense_all_negative() {
        // 2x2 matrix, every entry present, all negative.
        // Reduce along axis 0 (each column fully populated, size 2).
        let indices = vec![
            vec![0, 0], // col 0
            vec![1, 0], // col 0
            vec![0, 1], // col 1
            vec![1, 1], // col 1
        ];
        let values = vec![-1.0, -3.0, -5.0, -2.0];
        let coo = CooTensor::new(indices, values, vec![2, 2]).unwrap();

        let result = max_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[2]);

        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        // Column 0: max(-1, -3) = -1  (NOT 0)
        // Column 1: max(-5, -2) = -2  (NOT 0)
        assert_eq!(data[[0]], -1.0);
        assert_eq!(data[[1]], -2.0);
        // Both slices are genuinely nonzero maxima -> present in sparse result.
        assert_eq!(result.nnz(), 2);
    }

    /// Partially-populated slice with all-negative observed values: the
    /// structural (implicit) zero wins, so the max is 0. A sibling column
    /// that is fully populated keeps its true negative max.
    #[test]
    fn test_max_axis_partial_all_negative() {
        // 3x2 matrix, reduce along axis 0 (size 3).
        // Column 0: rows 0,1 present (partial) -> implicit zero present.
        // Column 1: rows 0,1,2 present (full)  -> no implicit zero.
        let indices = vec![
            vec![0, 0],
            vec![1, 0], // col 0 partial (2 of 3)
            vec![0, 1],
            vec![1, 1],
            vec![2, 1], // col 1 full (3 of 3)
        ];
        let values = vec![-1.0, -2.0, -4.0, -5.0, -6.0];
        let coo = CooTensor::new(indices, values, vec![3, 2]).unwrap();

        let result = max_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[2]);

        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        // Column 0: partial -> max(max(-1,-2), 0) = 0
        assert_eq!(data[[0]], 0.0);
        // Column 1: full -> max(-4,-5,-6) = -4
        assert_eq!(data[[1]], -4.0);
        // Only the fully-populated negative column survives as an entry.
        assert_eq!(result.nnz(), 1);
    }

    /// A slice with no observed entries: max is the implicit 0.
    #[test]
    fn test_max_axis_empty_slice() {
        // 2x2 matrix, reduce along axis 0. Column 1 has no entries.
        let indices = vec![
            vec![0, 0],
            vec![1, 0], // col 0 full, all negative
        ];
        let values = vec![-1.0, -2.0];
        let coo = CooTensor::new(indices, values, vec![2, 2]).unwrap();

        let result = max_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[2]);

        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        // Column 0: full negative -> max(-1,-2) = -1
        assert_eq!(data[[0]], -1.0);
        // Column 1: empty -> implicit 0
        assert_eq!(data[[1]], 0.0);
        assert_eq!(result.nnz(), 1);
    }

    // ------------------------------------------------------------------
    // Mirror tests for min_axis. min_axis already tracks the observed
    // count per slice, so it does NOT share the max_axis clamp bug. These
    // tests document and lock in that correct behavior.
    // ------------------------------------------------------------------

    /// Fully-dense, all-positive slices: min must be the smallest observed
    /// positive value, never the implicit 0 (there is no implicit zero).
    #[test]
    fn test_min_axis_fully_dense_all_positive() {
        // 2x2 matrix, every entry present, all positive.
        let indices = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let values = vec![1.0, 3.0, 5.0, 2.0];
        let coo = CooTensor::new(indices, values, vec![2, 2]).unwrap();

        let result = min_axis(&coo, 0).unwrap();
        assert_eq!(result.shape(), &[2]);

        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        // Column 0: min(1, 3) = 1  (NOT 0)
        // Column 1: min(5, 2) = 2  (NOT 0)
        assert_eq!(data[[0]], 1.0);
        assert_eq!(data[[1]], 2.0);
        assert_eq!(result.nnz(), 2);
    }

    /// Partially-populated all-positive slice: implicit zero wins -> min 0.
    #[test]
    fn test_min_axis_partial_all_positive() {
        // 3x2 matrix, reduce along axis 0 (size 3).
        let indices = vec![
            vec![0, 0],
            vec![1, 0], // col 0 partial (2 of 3)
            vec![0, 1],
            vec![1, 1],
            vec![2, 1], // col 1 full (3 of 3)
        ];
        let values = vec![1.0, 2.0, 4.0, 5.0, 6.0];
        let coo = CooTensor::new(indices, values, vec![3, 2]).unwrap();

        let result = min_axis(&coo, 0).unwrap();
        let dense = result.to_dense().unwrap();
        let data = dense.as_array();
        // Column 0: partial -> min(min(1,2), 0) = 0
        assert_eq!(data[[0]], 0.0);
        // Column 1: full -> min(4,5,6) = 4
        assert_eq!(data[[1]], 4.0);
        assert_eq!(result.nnz(), 1);
    }

    // ------------------------------------------------------------------
    // Randomized cross-check: max_axis / min_axis must agree with a
    // densify()-then-reduce reference over tensors of mixed sign and
    // mixed density (fully-dense, partial, and empty slices all appear).
    // ------------------------------------------------------------------

    #[test]
    fn test_max_min_axis_cross_check_dense_reference() {
        use scirs2_core::ndarray_ext::{Array, Axis, IxDyn};
        use scirs2_core::random::{SeedableRng, StdRng};

        /// Enumerate every multi-index of `shape` in row-major order.
        fn all_indices(shape: &[usize]) -> Vec<Vec<usize>> {
            let total: usize = shape.iter().product();
            let mut out = Vec::with_capacity(total);
            for flat in 0..total {
                let mut rem = flat;
                let mut idx = vec![0usize; shape.len()];
                for d in (0..shape.len()).rev() {
                    idx[d] = rem % shape[d];
                    rem /= shape[d];
                }
                out.push(idx);
            }
            out
        }

        /// Reference max over `axis` on a dense array (implicit zeros already
        /// materialized as explicit 0.0 by densification).
        fn dense_max_axis(dense: &Array<f64, IxDyn>, axis: usize) -> Array<f64, IxDyn> {
            dense.fold_axis(
                Axis(axis),
                f64::NEG_INFINITY,
                |&a, &b| if b > a { b } else { a },
            )
        }
        fn dense_min_axis(dense: &Array<f64, IxDyn>, axis: usize) -> Array<f64, IxDyn> {
            dense.fold_axis(
                Axis(axis),
                f64::INFINITY,
                |&a, &b| if b < a { b } else { a },
            )
        }

        let shapes: [Vec<usize>; 3] = [vec![4, 3], vec![2, 3, 4], vec![5, 5]];

        for (trial, shape) in shapes.iter().enumerate() {
            for seed in 0..4u64 {
                let mut rng = StdRng::seed_from_u64(seed + 1000 * trial as u64);

                // Build a dense reference array with mixed-sign, magnitude-bounded
                // values, sparsifying a random subset to implicit zeros.
                let mut dense = Array::<f64, _>::zeros(IxDyn(shape));
                let mut indices: Vec<Vec<usize>> = Vec::new();
                let mut values: Vec<f64> = Vec::new();

                for idx in all_indices(shape) {
                    // ~40% of entries are structural zeros -> guarantees a mix of
                    // fully-dense, partial, and (occasionally) empty slices.
                    let present = rng.random_f64() > 0.4;
                    if present {
                        let sign = if rng.random_f64() > 0.5 { 1.0 } else { -1.0 };
                        // Magnitude in [0.1, 1.0): safely above `epsilon`, so no
                        // near-zero values get dropped by the sparsity filter.
                        let mag = 0.1 + 0.9 * rng.random_f64();
                        let val = sign * mag;
                        dense[&idx[..]] = val;
                        indices.push(idx);
                        values.push(val);
                    }
                }

                let coo = CooTensor::new(indices, values, shape.clone()).unwrap();

                for axis in 0..shape.len() {
                    // ---- max_axis ----
                    let got_max = max_axis(&coo, axis).unwrap();
                    let got_max_dense = got_max.to_dense().unwrap();
                    let ref_max = dense_max_axis(&dense, axis);
                    assert_eq!(
                        got_max_dense.as_array().shape(),
                        ref_max.shape(),
                        "max_axis shape mismatch (shape={shape:?}, axis={axis}, seed={seed})"
                    );
                    for (g, r) in got_max_dense.as_array().iter().zip(ref_max.iter()) {
                        assert!(
                            (g - r).abs() < 1e-9,
                            "max_axis mismatch: got {g}, ref {r} (shape={shape:?}, axis={axis}, seed={seed})"
                        );
                    }

                    // ---- min_axis ----
                    let got_min = min_axis(&coo, axis).unwrap();
                    let got_min_dense = got_min.to_dense().unwrap();
                    let ref_min = dense_min_axis(&dense, axis);
                    for (g, r) in got_min_dense.as_array().iter().zip(ref_min.iter()) {
                        assert!(
                            (g - r).abs() < 1e-9,
                            "min_axis mismatch: got {g}, ref {r} (shape={shape:?}, axis={axis}, seed={seed})"
                        );
                    }
                }
            }
        }
    }
}
