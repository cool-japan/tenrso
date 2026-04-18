//! Shape manipulation helpers for the `TenrsoExecutor` implementation.
//!
//! Contains free functions supporting: `transpose`, `reshape`, `concatenate`, `split`,
//! `tile`, `pad`, `flip`, `squeeze`, `unsqueeze`, `stack`, `repeat`, `roll`.

use super::super::functions::TenrsoExecutor;
use super::super::types::CpuExecutor;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND, TensorHandle};

pub(super) fn transpose<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axes: &[Axis],
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for transpose"))?;
    let ndim = dense.shape().len();
    if axes.len() != ndim {
        return Err(anyhow!(
            "Axes length ({}) must match tensor dimensionality ({})",
            axes.len(),
            ndim
        ));
    }
    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim {
            return Err(anyhow!("Axis {} out of range for {}D tensor", axis, ndim));
        }
        if seen[axis] {
            return Err(anyhow!("Duplicate axis {} in permutation", axis));
        }
        seen[axis] = true;
    }
    let permuted = dense.view().permuted_axes(axes);
    let result = DenseND::from_array(permuted.to_owned());
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn reshape<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    new_shape: &[usize],
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for reshape"))?;
    let old_size: usize = dense.shape().iter().product();
    let new_size: usize = new_shape.iter().product();
    if old_size != new_size {
        return Err(anyhow!(
            "Cannot reshape tensor of size {} to size {}",
            old_size,
            new_size
        ));
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let data: Vec<T> = dense.view().iter().cloned().collect();
    let reshaped = Array::from_shape_vec(IxDyn(new_shape), data)
        .map_err(|e| anyhow!("Reshape failed: {}", e))?;
    let result = DenseND::from_array(reshaped);
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn concatenate<T>(
    executor: &mut CpuExecutor,
    tensors: &[TensorHandle<T>],
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    if tensors.is_empty() {
        return Err(anyhow!("Cannot concatenate empty tensor list"));
    }
    let dense_tensors: Vec<&DenseND<T>> = tensors
        .iter()
        .map(|t| {
            t.as_dense()
                .ok_or_else(|| anyhow!("Only dense tensors supported for concatenate"))
        })
        .collect::<Result<Vec<_>>>()?;
    let ndim = dense_tensors[0].shape().len();
    for t in dense_tensors.iter().skip(1) {
        if t.shape().len() != ndim {
            return Err(anyhow!(
                "All tensors must have same number of dimensions, got {} and {}",
                ndim,
                t.shape().len()
            ));
        }
    }
    if axis >= ndim {
        return Err(anyhow!("Axis {} out of range for {}D tensor", axis, ndim));
    }
    for dim in 0..ndim {
        if dim != axis {
            let expected_size = dense_tensors[0].shape()[dim];
            for (i, t) in dense_tensors.iter().enumerate().skip(1) {
                if t.shape()[dim] != expected_size {
                    return Err(anyhow!(
                        "Dimension {} mismatch: tensor 0 has size {}, tensor {} has size {}",
                        dim,
                        expected_size,
                        i,
                        t.shape()[dim]
                    ));
                }
            }
        }
    }
    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let mut output_shape = dense_tensors[0].shape().to_vec();
    for t in dense_tensors.iter().skip(1) {
        output_shape[axis] += t.shape()[axis];
    }
    let output_size: usize = output_shape.iter().product();

    // Use pooled buffer for concatenate output (Phase 5: Automatic Pooling)
    let mut output_data = executor.acquire_pooled_generic::<T>(&output_shape);
    output_data.clear(); // Ensure buffer starts empty
    output_data.reserve(output_size);

    for flat_idx in 0..output_size {
        let out_idx = executor.flat_to_multidim(flat_idx, &output_shape);
        let mut cumulative_axis_size = 0;
        let mut tensor_idx = 0;
        let mut local_axis_pos = out_idx[axis];
        for (i, t) in dense_tensors.iter().enumerate() {
            let t_axis_size = t.shape()[axis];
            if local_axis_pos < cumulative_axis_size + t_axis_size {
                tensor_idx = i;
                local_axis_pos -= cumulative_axis_size;
                break;
            }
            cumulative_axis_size += t_axis_size;
        }
        let mut src_idx = out_idx.clone();
        src_idx[axis] = local_axis_pos;
        let val = dense_tensors[tensor_idx].view()[src_idx.as_slice()];
        output_data.push(val);
    }

    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output_data.clone())
        .map_err(|e| anyhow!("Concatenation failed: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output_data);
    let result = DenseND::from_array(result_array);
    Ok(TensorHandle::from_dense_auto(result))
}

pub(super) fn split<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    num_splits: usize,
    axis: Axis,
) -> Result<Vec<TensorHandle<T>>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for split"))?;
    let ndim = dense.shape().len();
    if axis >= ndim {
        return Err(anyhow!("Axis {} out of range for {}D tensor", axis, ndim));
    }
    let axis_size = dense.shape()[axis];
    if axis_size % num_splits != 0 {
        return Err(anyhow!(
            "Cannot split axis of size {} into {} equal parts",
            axis_size,
            num_splits
        ));
    }
    let split_size = axis_size / num_splits;
    let mut results = Vec::with_capacity(num_splits);
    use scirs2_core::ndarray_ext::Axis as NdAxis;
    for i in 0..num_splits {
        let start = i * split_size;
        let end = start + split_size;
        let sliced = dense
            .view()
            .slice_axis(NdAxis(axis), (start..end).into())
            .to_owned();
        results.push(TensorHandle::from_dense_auto(DenseND::from_array(sliced)));
    }
    Ok(results)
}

pub(super) fn tile<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    reps: &[usize],
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for tile"))?;

    let input_shape = dense.shape();

    // Ensure reps has same length as input shape
    if reps.len() != input_shape.len() {
        return Err(anyhow!(
            "Reps length {} must match input dimensions {}",
            reps.len(),
            input_shape.len()
        ));
    }

    // Calculate output shape
    let output_shape: Vec<usize> = input_shape
        .iter()
        .zip(reps.iter())
        .map(|(&dim, &rep)| dim * rep)
        .collect();

    let input_view = dense.view();
    let output_size: usize = output_shape.iter().product();

    // Use pooled buffer for tile output (Phase 5: Automatic Pooling)
    let mut output_data = executor.acquire_pooled_generic::<T>(&output_shape);
    output_data.clear(); // Ensure buffer starts empty
    output_data.reserve(output_size);

    // Generate all output indices and map to input indices
    for i in 0..output_size {
        let out_idx = executor.flat_to_multidim(i, &output_shape);
        // Map output index to input index by taking modulo
        let in_idx: Vec<usize> = out_idx
            .iter()
            .zip(input_shape.iter())
            .map(|(&o, &s)| o % s)
            .collect();
        output_data.push(input_view[in_idx.as_slice()]);
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output_data.clone())
        .map_err(|e| anyhow!("Failed to create tiled array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output_data);

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn pad<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    pad_width: &[(usize, usize)],
    constant_value: T,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for pad"))?;

    let input_shape = dense.shape();

    if pad_width.len() != input_shape.len() {
        return Err(anyhow!(
            "Pad width length {} must match input dimensions {}",
            pad_width.len(),
            input_shape.len()
        ));
    }

    // Calculate output shape
    let output_shape: Vec<usize> = input_shape
        .iter()
        .zip(pad_width.iter())
        .map(|(&dim, &(before, after))| dim + before + after)
        .collect();

    let input_view = dense.view();
    let output_size: usize = output_shape.iter().product();

    // Use pooled buffer for pad output (Phase 5: Automatic Pooling)
    let mut output_data = executor.acquire_pooled_generic::<T>(&output_shape);
    output_data.clear(); // Ensure buffer starts empty
    output_data.resize(output_size, constant_value); // Initialize with constant_value

    // Copy input data to the appropriate region in output
    let input_size: usize = input_shape.iter().product();
    for i in 0..input_size {
        let in_idx = executor.flat_to_multidim(i, input_shape);
        // Calculate output index by adding padding offsets
        let out_idx: Vec<usize> = in_idx
            .iter()
            .zip(pad_width.iter())
            .map(|(&idx, &(before, _))| idx + before)
            .collect();
        let out_flat = executor.multidim_to_flat(&out_idx, &output_shape);
        output_data[out_flat] = input_view[in_idx.as_slice()];
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output_data.clone())
        .map_err(|e| anyhow!("Failed to create padded array: {}", e))?;
    executor.release_pooled_generic::<T>(&output_shape, output_data);

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn flip<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axes: &[Axis],
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for flip"))?;

    let shape = dense.shape();

    // Validate axes
    for &axis in axes {
        if axis >= shape.len() {
            return Err(anyhow!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            ));
        }
    }

    let input_view = dense.view();
    let total_elements: usize = shape.iter().product();

    // Use pooled buffer for flip output (Phase 5: Automatic Pooling)
    let mut output_data = executor.acquire_pooled_generic::<T>(shape);
    output_data.clear(); // Ensure buffer starts empty
    output_data.reserve(total_elements);

    // For each output position, compute the corresponding flipped input position
    for i in 0..total_elements {
        let out_idx = executor.flat_to_multidim(i, shape);
        // Flip specified axes
        let in_idx: Vec<usize> = out_idx
            .iter()
            .enumerate()
            .map(|(axis, &idx)| {
                if axes.contains(&axis) {
                    // Flip this axis
                    shape[axis] - 1 - idx
                } else {
                    idx
                }
            })
            .collect();
        output_data.push(input_view[in_idx.as_slice()]);
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(shape), output_data.clone())
        .map_err(|e| anyhow!("Failed to create flipped array: {}", e))?;
    executor.release_pooled_generic::<T>(shape, output_data);

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn squeeze<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axes: Option<&[Axis]>,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for squeeze"))?;

    let shape = dense.shape();

    // Determine which axes to squeeze
    let axes_to_squeeze: Vec<usize> = if let Some(ax) = axes {
        // Validate and collect specified axes
        for &axis in ax {
            if axis >= shape.len() {
                return Err(anyhow!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    shape.len()
                ));
            }
            if shape[axis] != 1 {
                return Err(anyhow!(
                    "Cannot squeeze axis {} with size {}",
                    axis,
                    shape[axis]
                ));
            }
        }
        ax.to_vec()
    } else {
        // Find all axes with size 1
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s == 1 { Some(i) } else { None })
            .collect()
    };

    // Build new shape by removing squeezed axes
    let new_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| {
            if axes_to_squeeze.contains(&i) {
                None
            } else {
                Some(s)
            }
        })
        .collect();

    // If no dimensions to squeeze, return original
    if new_shape.len() == shape.len() {
        return Ok(x.clone());
    }

    // Reshape - data order is preserved
    executor.reshape(x, &new_shape)
}

pub(super) fn unsqueeze<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for unsqueeze"))?;

    let shape = dense.shape();

    // axis can be at most shape.len() (to append at the end)
    if axis > shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for unsqueeze (max {})",
            axis,
            shape.len()
        ));
    }

    // Build new shape by inserting 1 at the specified axis
    let mut new_shape = shape.to_vec();
    new_shape.insert(axis, 1);

    executor.reshape(x, &new_shape)
}

pub(super) fn stack<T>(
    executor: &mut CpuExecutor,
    tensors: &[TensorHandle<T>],
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    if tensors.is_empty() {
        return Err(anyhow!("Cannot stack empty sequence of tensors"));
    }

    // Get shape of first tensor
    let first_shape = tensors[0]
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for stack"))?
        .shape();

    // Validate all tensors have same shape
    for (i, tensor) in tensors.iter().enumerate().skip(1) {
        let shape = tensor
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for stack"))?
            .shape();
        if shape != first_shape {
            return Err(anyhow!(
                "All tensors must have the same shape for stacking. Tensor 0: {:?}, Tensor {}: {:?}",
                first_shape,
                i,
                shape
            ));
        }
    }

    if axis > first_shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for stack (max {})",
            axis,
            first_shape.len()
        ));
    }

    // Unsqueeze all tensors at the specified axis
    let mut unsqueezed = Vec::new();
    for tensor in tensors {
        unsqueezed.push(executor.unsqueeze(tensor, axis)?);
    }

    // Concatenate along the new axis
    executor.concatenate(&unsqueezed, axis)
}

pub(super) fn repeat<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    repeats: usize,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for repeat"))?;

    let shape = dense.shape();

    if axis >= shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        ));
    }

    if repeats == 0 {
        return Err(anyhow!("Repeat count must be greater than 0"));
    }

    if repeats == 1 {
        return Ok(x.clone());
    }

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    output_shape[axis] *= repeats;

    let input_view = dense.view();
    let total_elements: usize = output_shape.iter().product();
    let mut output_data = Vec::with_capacity(total_elements);

    // Generate output by repeating each element along the specified axis
    for i in 0..total_elements {
        let out_idx = executor.flat_to_multidim(i, &output_shape);
        // Map output index to input index by dividing by repeats
        let mut in_idx = out_idx.clone();
        in_idx[axis] /= repeats;
        output_data.push(input_view[in_idx.as_slice()]);
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(&output_shape), output_data)
        .map_err(|e| anyhow!("Failed to create repeated array: {}", e))?;

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn roll<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    shift: isize,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for roll"))?;

    let shape = dense.shape();

    if axis >= shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        ));
    }

    if shift == 0 {
        return Ok(x.clone());
    }

    let axis_size = shape[axis] as isize;
    // Normalize shift to [0, axis_size)
    let normalized_shift = ((shift % axis_size) + axis_size) % axis_size;

    if normalized_shift == 0 {
        return Ok(x.clone());
    }

    let input_view = dense.view();
    let total_elements: usize = shape.iter().product();
    let mut output_data = Vec::with_capacity(total_elements);

    // Generate output by rolling indices along the specified axis
    for i in 0..total_elements {
        let out_idx = executor.flat_to_multidim(i, shape);
        // Calculate rolled index for the axis
        let mut in_idx = out_idx.clone();
        let old_idx = out_idx[axis] as isize;
        let new_idx = ((old_idx - normalized_shift + axis_size) % axis_size) as usize;
        in_idx[axis] = new_idx;
        output_data.push(input_view[in_idx.as_slice()]);
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_array = Array::from_shape_vec(IxDyn(shape), output_data)
        .map_err(|e| anyhow!("Failed to create rolled array: {}", e))?;

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}
