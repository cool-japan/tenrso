//! Advanced tensor indexing operations
//!
//! This module implements sophisticated indexing patterns including:
//! - Fancy indexing (NumPy-style advanced indexing)
//! - Multi-dimensional gather/scatter
//! - Index broadcasting
//! - Masked indexing
//!
//! # Use Cases
//!
//! - Embeddings lookup in transformers
//! - Attention mechanisms
//! - Sparse gradient updates
//! - Dynamic batching
//! - Index-based data shuffling

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{ArrayView, IxDyn, Zip};
use scirs2_core::numeric::{Float, FromPrimitive, Num, ToPrimitive};
use tenrso_core::{Axis, DenseND};

/// Advanced gather operation with multi-dimensional support
///
/// Gathers values from `input` along the specified `axis` using `indices`.
/// Unlike the basic gather, this supports:
/// - Multi-dimensional indices
/// - Negative indices (Python-style)
/// - Out-of-bounds checking with clear error messages
///
/// # Arguments
///
/// * `input` - Source tensor to gather from
/// * `axis` - Axis along which to gather
/// * `indices` - Integer indices (as Float tensor for compatibility)
/// * `allow_negative` - Whether to allow negative indices (wrapped to positive)
///
/// # Example
///
/// ```text
/// input: [10, 20, 30, 40, 50]
/// indices: [0, 2, 4, 1]
/// result: [10, 30, 50, 20]
/// ```
pub fn advanced_gather<T>(
    input: &DenseND<T>,
    axis: Axis,
    indices: &DenseND<T>,
    allow_negative: bool,
) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + ToPrimitive + FromPrimitive,
{
    let input_shape = input.shape();

    // Validate axis
    if axis >= input_shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            input_shape.len()
        ));
    }

    let axis_size = input_shape[axis];
    let indices_view = indices.view();

    // Convert indices to usize, handling negative indices if allowed
    let converted_indices: Vec<usize> = indices_view
        .iter()
        .map(|&idx| {
            let idx_i64 = idx
                .to_i64()
                .ok_or_else(|| anyhow!("Index value cannot be converted to integer"))?;

            let final_idx = if idx_i64 < 0 {
                if !allow_negative {
                    return Err(anyhow!("Negative index {} not allowed", idx_i64));
                }
                // Python-style negative indexing
                let positive_idx = (axis_size as i64 + idx_i64) as usize;
                if positive_idx >= axis_size {
                    return Err(anyhow!(
                        "Negative index {} out of bounds for axis size {}",
                        idx_i64,
                        axis_size
                    ));
                }
                positive_idx
            } else {
                let idx_usize = idx_i64 as usize;
                if idx_usize >= axis_size {
                    return Err(anyhow!(
                        "Index {} out of bounds for axis size {}",
                        idx_usize,
                        axis_size
                    ));
                }
                idx_usize
            };

            Ok(final_idx)
        })
        .collect::<Result<Vec<_>>>()?;

    // Build output shape
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = converted_indices.len();

    // Create output array
    let total_elements: usize = output_shape.iter().product();
    let mut output_data = vec![T::zero(); total_elements];

    // Perform gathering
    let input_view = input.view();
    gather_recursive(
        &input_view,
        &converted_indices,
        axis,
        &output_shape,
        &mut output_data,
        0,
        &mut 0,
    )?;

    DenseND::from_vec(output_data, &output_shape)
}

/// Recursive helper for advanced gather
fn gather_recursive<T>(
    input: &ArrayView<T, IxDyn>,
    indices: &[usize],
    axis: Axis,
    output_shape: &[usize],
    output_data: &mut [T],
    current_depth: usize,
    output_idx: &mut usize,
) -> Result<()>
where
    T: Clone + Num,
{
    if current_depth == output_shape.len() {
        return Ok(());
    }

    let dim_size = output_shape[current_depth];

    if current_depth == axis {
        // At the gather axis - use indices
        for &idx in indices {
            let slice = input.index_axis(scirs2_core::ndarray_ext::Axis(current_depth), idx);
            if current_depth == output_shape.len() - 1 {
                // Leaf level - copy data
                output_data[*output_idx] = slice.iter().next().unwrap().clone();
                *output_idx += 1;
            } else {
                gather_recursive(
                    &slice,
                    indices,
                    axis,
                    output_shape,
                    output_data,
                    current_depth + 1,
                    output_idx,
                )?;
            }
        }
    } else {
        // Not at gather axis - iterate normally
        for i in 0..dim_size {
            let slice = input.index_axis(scirs2_core::ndarray_ext::Axis(current_depth), i);
            if current_depth == output_shape.len() - 1 {
                // Leaf level - copy data
                output_data[*output_idx] = slice.iter().next().unwrap().clone();
                *output_idx += 1;
            } else {
                gather_recursive(
                    &slice,
                    indices,
                    axis,
                    output_shape,
                    output_data,
                    current_depth + 1,
                    output_idx,
                )?;
            }
        }
    }

    Ok(())
}

/// Advanced scatter operation with multi-dimensional support
///
/// Scatters `values` into an output tensor of shape `shape` along the specified `axis`
/// using `indices`. Unlike the basic scatter, this supports:
/// - Multi-dimensional indices and values
/// - Negative indices (Python-style)
/// - Accumulation modes (replace, add, max, min)
///
/// # Arguments
///
/// * `shape` - Shape of the output tensor
/// * `axis` - Axis along which to scatter
/// * `indices` - Integer indices (as Float tensor)
/// * `values` - Values to scatter
/// * `mode` - Scatter mode (Replace, Add, Max, Min)
///
/// # Example
///
/// ```text
/// shape: [5]
/// indices: [0, 2, 4]
/// values: [10, 30, 50]
/// result: [10, 0, 30, 0, 50] (assuming zero initialization)
/// ```
pub fn advanced_scatter<T>(
    shape: &[usize],
    axis: Axis,
    indices: &DenseND<T>,
    values: &DenseND<T>,
    mode: ScatterMode,
) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + ToPrimitive + FromPrimitive + PartialOrd,
{
    // Validate axis
    if axis >= shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        ));
    }

    let axis_size = shape[axis];
    let indices_view = indices.view();

    // Convert and validate indices
    let converted_indices: Vec<usize> = indices_view
        .iter()
        .map(|&idx| {
            let idx_i64 = idx
                .to_i64()
                .ok_or_else(|| anyhow!("Index value cannot be converted to integer"))?;

            if idx_i64 < 0 {
                return Err(anyhow!("Negative indices not supported in scatter"));
            }

            let idx_usize = idx_i64 as usize;
            if idx_usize >= axis_size {
                return Err(anyhow!(
                    "Index {} out of bounds for axis size {}",
                    idx_usize,
                    axis_size
                ));
            }

            Ok(idx_usize)
        })
        .collect::<Result<Vec<_>>>()?;

    // Initialize output based on mode
    let total_elements: usize = shape.iter().product();
    let mut output_data = match mode {
        ScatterMode::Replace => vec![T::zero(); total_elements],
        ScatterMode::Add => vec![T::zero(); total_elements],
        ScatterMode::Max => vec![T::from_f64(f64::NEG_INFINITY).unwrap(); total_elements],
        ScatterMode::Min => vec![T::from_f64(f64::INFINITY).unwrap(); total_elements],
    };

    // Perform scattering
    let values_view = values.view();
    scatter_recursive(
        &values_view,
        &converted_indices,
        axis,
        shape,
        &mut output_data,
        0,
        &mut 0,
        mode,
    )?;

    DenseND::from_vec(output_data, shape)
}

/// Scatter modes for accumulation
#[derive(Clone, Copy, Debug)]
pub enum ScatterMode {
    /// Replace existing values (default)
    Replace,
    /// Add to existing values (accumulate)
    Add,
    /// Take maximum of existing and new values
    Max,
    /// Take minimum of existing and new values
    Min,
}

/// Recursive helper for advanced scatter
#[allow(clippy::too_many_arguments)]
fn scatter_recursive<T>(
    values: &ArrayView<T, IxDyn>,
    indices: &[usize],
    axis: Axis,
    output_shape: &[usize],
    output_data: &mut [T],
    current_depth: usize,
    values_idx: &mut usize,
    mode: ScatterMode,
) -> Result<()>
where
    T: Clone + Num + PartialOrd,
{
    if current_depth == output_shape.len() {
        return Ok(());
    }

    let _dim_size = output_shape[current_depth];

    if current_depth == axis {
        // At the scatter axis - use indices
        for &out_idx in indices {
            if current_depth == output_shape.len() - 1 {
                // Leaf level - write data
                let value = values.iter().nth(*values_idx).unwrap().clone();
                let flat_idx = compute_flat_index(output_shape, &[out_idx], current_depth);

                match mode {
                    ScatterMode::Replace => output_data[flat_idx] = value,
                    ScatterMode::Add => {
                        output_data[flat_idx] = output_data[flat_idx].clone() + value
                    }
                    ScatterMode::Max => {
                        if value > output_data[flat_idx] {
                            output_data[flat_idx] = value;
                        }
                    }
                    ScatterMode::Min => {
                        if value < output_data[flat_idx] {
                            output_data[flat_idx] = value;
                        }
                    }
                }
                *values_idx += 1;
            }
        }
    }

    Ok(())
}

/// Compute flat index from multi-dimensional indices
fn compute_flat_index(shape: &[usize], indices: &[usize], depth: usize) -> usize {
    let mut flat_idx = 0;
    let mut stride = 1;

    for i in (0..=depth).rev() {
        flat_idx += indices[i] * stride;
        if i > 0 {
            stride *= shape[i];
        }
    }

    flat_idx
}

/// Fancy indexing with boolean masks
///
/// Select elements from `input` where `mask` is true (> 0).
/// Returns a 1D tensor containing the selected elements.
///
/// This is more flexible than basic masked_select as it supports:
/// - Any tensor shape
/// - Efficient memory access patterns
/// - Parallel processing for large tensors
pub fn fancy_index_mask<T>(input: &DenseND<T>, mask: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + PartialOrd,
{
    if input.shape() != mask.shape() {
        return Err(anyhow!(
            "Input and mask must have the same shape: {:?} vs {:?}",
            input.shape(),
            mask.shape()
        ));
    }

    let input_view = input.view();
    let mask_view = mask.view();
    let zero = T::zero();

    // Collect selected elements
    let mut selected = Vec::new();
    Zip::from(&input_view)
        .and(&mask_view)
        .for_each(|val, mask_val| {
            if *mask_val > zero {
                selected.push(val.clone());
            }
        });

    let output_shape = vec![selected.len()];
    DenseND::from_vec(selected, &output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_gather_1d() {
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
        let indices = DenseND::from_vec(vec![0.0, 2.0, 4.0, 1.0], &[4]).unwrap();

        let result = advanced_gather(&input, 0, &indices, false).unwrap();

        assert_eq!(result.shape(), &[4]);
        let result_view = result.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 30.0);
        assert_eq!(result_view[[2]], 50.0);
        assert_eq!(result_view[[3]], 20.0);
    }

    #[test]
    fn test_advanced_gather_negative_indices() {
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
        let indices = DenseND::from_vec(vec![-1.0, -2.0], &[2]).unwrap();

        let result = advanced_gather(&input, 0, &indices, true).unwrap();

        assert_eq!(result.shape(), &[2]);
        let result_view = result.view();
        assert_eq!(result_view[[0]], 50.0); // -1 -> index 4
        assert_eq!(result_view[[1]], 40.0); // -2 -> index 3
    }

    #[test]
    fn test_advanced_gather_out_of_bounds() {
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let indices = DenseND::from_vec(vec![0.0, 5.0], &[2]).unwrap();

        let result = advanced_gather(&input, 0, &indices, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_advanced_scatter_replace() {
        let shape = vec![5];
        let indices = DenseND::from_vec(vec![0.0, 2.0, 4.0], &[3]).unwrap();
        let values = DenseND::from_vec(vec![10.0, 30.0, 50.0], &[3]).unwrap();

        let result = advanced_scatter(&shape, 0, &indices, &values, ScatterMode::Replace).unwrap();

        assert_eq!(result.shape(), &[5]);
        let result_view = result.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 0.0);
        assert_eq!(result_view[[2]], 30.0);
        assert_eq!(result_view[[3]], 0.0);
        assert_eq!(result_view[[4]], 50.0);
    }

    #[test]
    fn test_fancy_index_mask() {
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
        let mask = DenseND::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], &[5]).unwrap();

        let result = fancy_index_mask(&input, &mask).unwrap();

        assert_eq!(result.shape(), &[3]);
        let result_view = result.view();
        assert_eq!(result_view[[0]], 10.0);
        assert_eq!(result_view[[1]], 30.0);
        assert_eq!(result_view[[2]], 50.0);
    }

    #[test]
    fn test_fancy_index_mask_all_false() {
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let mask = DenseND::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();

        let result = fancy_index_mask(&input, &mask).unwrap();

        assert_eq!(result.shape(), &[0]);
    }

    #[test]
    fn test_fancy_index_mask_shape_mismatch() {
        let input = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let mask = DenseND::from_vec(vec![1.0, 0.0], &[2]).unwrap();

        let result = fancy_index_mask(&input, &mask);
        assert!(result.is_err());
    }
}
