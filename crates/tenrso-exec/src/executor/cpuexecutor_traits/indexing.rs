//! Indexing helpers for the `TenrsoExecutor` implementation.
//!
//! Contains free functions supporting: `where_op`, `masked_select`, `gather`, `scatter`,
//! `advanced_gather`, `advanced_scatter`, `fancy_index_mask`.

use super::super::types::{CpuExecutor, ScatterMode};
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND, TensorHandle};

pub(super) fn where_op<T>(
    _executor: &mut CpuExecutor,
    condition: &TensorHandle<T>,
    x: &TensorHandle<T>,
    y: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let cond_dense = condition
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for where_op"))?;
    let x_dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for where_op"))?;
    let y_dense = y
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for where_op"))?;
    if cond_dense.shape() != x_dense.shape() || x_dense.shape() != y_dense.shape() {
        return Err(anyhow!(
            "Shape mismatch: condition={:?}, x={:?}, y={:?}",
            cond_dense.shape(),
            x_dense.shape(),
            y_dense.shape()
        ));
    }
    use scirs2_core::ndarray_ext::Zip;
    let mut result = x_dense.view().to_owned();
    Zip::from(&mut result)
        .and(&cond_dense.view())
        .and(&x_dense.view())
        .and(&y_dense.view())
        .for_each(|r, &c, &x_val, &y_val| {
            *r = if c > T::zero() { x_val } else { y_val };
        });
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(result)))
}

pub(super) fn masked_select<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    mask: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let x_dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for masked_select"))?;
    let mask_dense = mask
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for masked_select"))?;
    if x_dense.shape() != mask_dense.shape() {
        return Err(anyhow!(
            "Shape mismatch: x={:?}, mask={:?}",
            x_dense.shape(),
            mask_dense.shape()
        ));
    }
    let mut selected = Vec::new();
    for (x_val, mask_val) in x_dense.view().iter().zip(mask_dense.view().iter()) {
        if *mask_val > T::zero() {
            selected.push(*x_val);
        }
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&[selected.len()]), selected)
        .map_err(|e| anyhow!("Failed to create result array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn gather<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
    indices: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_x = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for gather"))?;
    let dense_indices = indices
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for indices"))?;
    let x_shape = dense_x.shape();
    let axis_idx = axis;
    if axis_idx >= x_shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis_idx,
            x_shape.len()
        ));
    }
    let indices_shape = dense_indices.shape();
    let num_indices: usize = indices_shape.iter().product();
    let mut output_shape = Vec::new();
    for (i, &dim) in x_shape.iter().enumerate() {
        if i == axis_idx {
            output_shape.extend_from_slice(indices_shape);
        } else if i != axis_idx {
            output_shape.push(dim);
        }
    }
    let output_size: usize = output_shape.iter().product();
    let mut output = Vec::with_capacity(output_size);
    let x_view = dense_x.view();
    let indices_view = dense_indices.view();
    if axis_idx == 0 && indices_shape.len() == 1 {
        let axis_size = x_shape[0];
        let elements_per_item: usize = x_shape[1..].iter().product();
        for idx_flat in 0..num_indices {
            let idx_multi = executor.flat_to_multidim(idx_flat, indices_shape);
            let idx_val = indices_view[idx_multi.as_slice()];
            let idx = idx_val
                .to_usize()
                .ok_or_else(|| anyhow!("Invalid index value: cannot convert to usize"))?;
            if idx >= axis_size {
                return Err(anyhow!(
                    "Index {} out of bounds for axis {} with size {}",
                    idx,
                    axis_idx,
                    axis_size
                ));
            }
            for elem_idx in 0..elements_per_item {
                let mut x_index = vec![idx];
                let elem_multi = executor.flat_to_multidim(elem_idx, &x_shape[1..]);
                x_index.extend(elem_multi);
                output.push(x_view[x_index.as_slice()]);
            }
        }
    } else {
        return Err(anyhow!(
            "Gather only supports axis=0 with 1D indices in this implementation"
        ));
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn scatter<T>(
    executor: &mut CpuExecutor,
    shape: &[usize],
    axis: Axis,
    indices: &TensorHandle<T>,
    values: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense_indices = indices
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for indices"))?;
    let dense_values = values
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for values"))?;
    let axis_idx = axis;
    if axis_idx >= shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for output shape with {} dimensions",
            axis_idx,
            shape.len()
        ));
    }
    let indices_shape = dense_indices.shape();
    let values_shape = dense_values.shape();
    let num_indices: usize = indices_shape.iter().product();
    let mut expected_values_shape = Vec::new();
    expected_values_shape.extend_from_slice(&shape[..axis_idx]);
    expected_values_shape.extend_from_slice(indices_shape);
    expected_values_shape.extend_from_slice(&shape[axis_idx + 1..]);
    if values_shape != expected_values_shape.as_slice() {
        return Err(anyhow!(
            "Values shape {:?} doesn't match expected shape {:?}",
            values_shape,
            expected_values_shape
        ));
    }
    let output_size: usize = shape.iter().product();
    let mut output = vec![T::zero(); output_size];
    let indices_view = dense_indices.view();
    let values_view = dense_values.view();
    if axis_idx == 0 && indices_shape.len() == 1 {
        let axis_size = shape[0];
        let elements_per_item: usize = shape[1..].iter().product();
        for idx_flat in 0..num_indices {
            let idx_multi = executor.flat_to_multidim(idx_flat, indices_shape);
            let idx_val = indices_view[idx_multi.as_slice()];
            let idx = idx_val
                .to_usize()
                .ok_or_else(|| anyhow!("Invalid index value: cannot convert to usize"))?;
            if idx >= axis_size {
                return Err(anyhow!(
                    "Index {} out of bounds for axis {} with size {}",
                    idx,
                    axis_idx,
                    axis_size
                ));
            }
            for elem_idx in 0..elements_per_item {
                let mut values_index = vec![idx_flat];
                let elem_multi = executor.flat_to_multidim(elem_idx, &shape[1..]);
                values_index.extend(elem_multi.clone());
                let mut out_index = vec![idx];
                out_index.extend(elem_multi);
                let out_flat = executor.multidim_to_flat(&out_index, shape);
                output[out_flat] = values_view[values_index.as_slice()];
            }
        }
    } else {
        return Err(anyhow!(
            "Scatter only supports axis=0 with 1D indices in this implementation"
        ));
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(shape), output)
        .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn advanced_gather<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
    indices: &TensorHandle<T>,
    allow_negative: bool,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for advanced_gather"))?;
    let indices_dense = indices
        .as_dense()
        .ok_or_else(|| anyhow!("Indices must be dense tensor"))?;

    let result = super::super::advanced_indexing::advanced_gather(
        dense,
        axis,
        indices_dense,
        allow_negative,
    )?;
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn advanced_scatter<T>(
    _executor: &mut CpuExecutor,
    shape: &[usize],
    axis: Axis,
    indices: &TensorHandle<T>,
    values: &TensorHandle<T>,
    mode: ScatterMode,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let indices_dense = indices
        .as_dense()
        .ok_or_else(|| anyhow!("Indices must be dense tensor"))?;
    let values_dense = values
        .as_dense()
        .ok_or_else(|| anyhow!("Values must be dense tensor"))?;

    let result = super::super::advanced_indexing::advanced_scatter(
        shape,
        axis,
        indices_dense,
        values_dense,
        mode,
    )?;
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn fancy_index_mask<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    mask: &TensorHandle<T>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for fancy_index_mask"))?;
    let mask_dense = mask
        .as_dense()
        .ok_or_else(|| anyhow!("Mask must be dense tensor"))?;

    let result = super::super::advanced_indexing::fancy_index_mask(dense, mask_dense)?;
    Ok(TensorHandle::from_dense_auto(result))
}
