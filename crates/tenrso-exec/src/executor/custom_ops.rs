//! Custom operations with user-defined functions
//!
//! This module provides support for custom operations where users can define
//! their own reduction functions, element-wise operations, and more.

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND, TensorHandle};

/// Custom reduction operation with user-defined reduction function
///
/// # Arguments
/// * `input` - Input tensor
/// * `axes` - Axes along which to reduce (empty = reduce all)
/// * `init_value` - Initial value for the reduction
/// * `reduce_fn` - Binary reduction function: (accumulator, element) -> new_accumulator
///
/// # Example
/// ```ignore
/// // Compute product along axis 0
/// custom_reduce(&tensor, &[0], 1.0, |acc, x| acc * x)?;
/// ```
pub fn custom_reduce<T, F>(
    input: &DenseND<T>,
    axes: &[Axis],
    init_value: T,
    reduce_fn: F,
) -> Result<DenseND<T>>
where
    T: Clone + Num + std::ops::AddAssign,
    F: Fn(T, T) -> T,
{
    if axes.is_empty() {
        // Reduce all axes
        let input_view = input.view();
        let result = input_view.iter().cloned().fold(init_value, &reduce_fn);
        let result_array = Array::from_elem(IxDyn(&[]), result);
        return Ok(DenseND::from_array(result_array));
    }

    // Reduce along specific axes
    let mut result = input.clone();
    for &axis in axes {
        if axis >= result.shape().len() {
            return Err(anyhow::anyhow!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                result.shape().len()
            ));
        }

        let result_view = result.view();

        // Manual reduction along the specified axis
        let new_shape: Vec<usize> = result
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let axis_size = result.shape()[axis];
        let output_size: usize = new_shape.iter().product();
        let mut output_data = Vec::with_capacity(output_size);

        // Iterate over all output positions
        for out_idx in 0..output_size {
            let mut acc = init_value.clone();

            // Reduce along the specified axis
            for axis_idx in 0..axis_size {
                // Convert flat output index to multi-dimensional index
                let mut in_idx = Vec::with_capacity(result.shape().len());
                let mut remaining = out_idx;

                for (dim_idx, &_dim_size) in result.shape().iter().enumerate() {
                    if dim_idx == axis {
                        in_idx.push(axis_idx);
                    } else {
                        let stride: usize = new_shape
                            [if dim_idx < axis { dim_idx } else { dim_idx - 1 }..]
                            .iter()
                            .product();
                        in_idx.push(remaining / stride);
                        remaining %= stride;
                    }
                }

                let value = result_view[in_idx.as_slice()].clone();
                acc = reduce_fn(acc, value);
            }

            output_data.push(acc);
        }

        let result_array = Array::from_shape_vec(IxDyn(&new_shape), output_data)
            .map_err(|e| anyhow::anyhow!("Failed to create result array: {}", e))?;
        result = DenseND::from_array(result_array);
    }

    Ok(result)
}

/// Custom element-wise binary operation with user-defined function
///
/// Applies a custom binary operation element-wise to two tensors with broadcasting support.
///
/// # Arguments
/// * `x` - First input tensor
/// * `y` - Second input tensor
/// * `op_fn` - Binary operation function: (x_elem, y_elem) -> result_elem
///
/// # Example
/// ```ignore
/// // Custom operation: (x + y) / 2
/// custom_binary_op(&x, &y, |a, b| (a + b) / 2.0)?;
/// ```
pub fn custom_binary_op<T, F>(x: &DenseND<T>, y: &DenseND<T>, op_fn: F) -> Result<DenseND<T>>
where
    T: Clone + Num,
    F: Fn(T, T) -> T,
{
    let x_view = x.view();
    let y_view = y.view();

    if x.shape() == y.shape() {
        // Same shape - direct element-wise operation
        let result_data: Vec<T> = x_view
            .iter()
            .zip(y_view.iter())
            .map(|(a, b)| op_fn(a.clone(), b.clone()))
            .collect();

        let result_array = Array::from_shape_vec(IxDyn(x.shape()), result_data)
            .map_err(|e| anyhow::anyhow!("Failed to create result array: {}", e))?;
        return Ok(DenseND::from_array(result_array));
    }

    // Broadcasting case - simplified implementation
    // TODO: Implement full broadcasting support
    Err(anyhow::anyhow!(
        "Custom binary operations with broadcasting not yet implemented. Shapes: {:?} vs {:?}",
        x.shape(),
        y.shape()
    ))
}

/// Custom element-wise unary operation with user-defined function
///
/// # Arguments
/// * `input` - Input tensor
/// * `op_fn` - Unary operation function: (element) -> result_element
///
/// # Example
/// ```ignore
/// // Custom operation: sigmoid-like function
/// custom_unary_op(&input, |x| x / (1.0 + x.abs()))?;
/// ```
pub fn custom_unary_op<T, F>(input: &DenseND<T>, op_fn: F) -> Result<DenseND<T>>
where
    T: Clone + Num,
    F: Fn(T) -> T,
{
    let input_view = input.view();
    let result = input_view.mapv(op_fn);
    Ok(DenseND::from_array(result))
}

/// Apply a custom operation to a tensor handle
pub fn apply_custom_unary<T, F>(input: &TensorHandle<T>, op_fn: F) -> Result<TensorHandle<T>>
where
    T: Clone + Num + Float + FromPrimitive,
    F: Fn(T) -> T,
{
    if let Some(dense) = input.as_dense() {
        let result = custom_unary_op(dense, op_fn)?;
        Ok(TensorHandle::from_dense_auto(result))
    } else {
        Err(anyhow::anyhow!(
            "Custom operations only supported for dense tensors"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_reduce_product() {
        let input = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        // Product of all elements: 2 * 3 * 4 * 5 = 120
        let result = custom_reduce(&input, &[], 1.0, |acc, x| acc * x).unwrap();
        let result_view = result.view();

        assert!((result_view[[]] as f64 - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_reduce_max() {
        let input = DenseND::from_vec(vec![2.0, 8.0, 4.0, 5.0], &[4]).unwrap();
        // Max of all elements: 8.0
        let result = custom_reduce(&input, &[], f64::NEG_INFINITY, |acc, x| {
            if x > acc {
                x
            } else {
                acc
            }
        })
        .unwrap();
        let result_view = result.view();

        assert!((result_view[[]] as f64 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_unary_op() {
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        // Square each element
        let result = custom_unary_op(&input, |x| x * x).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] as f64 - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] as f64 - 4.0).abs() < 1e-10);
        assert!((result_view[[2]] as f64 - 9.0).abs() < 1e-10);
        assert!((result_view[[3]] as f64 - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_binary_op() {
        let x = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let y = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        // Custom operation: (x + y) / 2
        let result = custom_binary_op(&x, &y, |a, b| (a + b) / 2.0).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] as f64 - 1.5).abs() < 1e-10);
        assert!((result_view[[1]] as f64 - 2.5).abs() < 1e-10);
        assert!((result_view[[2]] as f64 - 3.5).abs() < 1e-10);
        assert!((result_view[[3]] as f64 - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_custom_unary() {
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let handle = TensorHandle::from_dense_auto(input);

        // Apply custom sigmoid-like function
        let result = apply_custom_unary(&handle, |x: f64| x / (1.0 + x.abs())).unwrap();
        let result_dense = result.as_dense().unwrap();
        let result_view = result_dense.view();

        // Check that values are in expected range (0, 1)
        for i in 0..4 {
            let val = result_view[[i]] as f64;
            assert!(val > 0.0 && val < 1.0);
        }
    }
}
