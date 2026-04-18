//! Reduction / normalization / arg* helpers for the `TenrsoExecutor` implementation.
//!
//! Contains free functions supporting: `reduce`, `softmax`, `log_softmax`,
//! `layer_norm`, `batch_norm`, `argmax`, `argmin`.

use super::super::types::{CpuExecutor, ReduceOp};
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND, TensorHandle};

pub(super) fn reduce<T>(
    _executor: &mut CpuExecutor,
    op: ReduceOp,
    x: &TensorHandle<T>,
    axes: &[Axis],
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for reduce"))?;
    if axes.is_empty() {
        return Err(anyhow!("No axes specified for reduction"));
    }
    let axis_indices: Vec<usize> = axes.to_vec();
    let ndim = dense.shape().len();
    for &axis_idx in &axis_indices {
        if axis_idx >= ndim {
            return Err(anyhow!(
                "Axis index {} out of range for tensor with {} dimensions",
                axis_idx,
                ndim
            ));
        }
    }
    let mut result = dense.view().to_owned();
    let mut sorted_axes = axis_indices.clone();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    for &axis_idx in &sorted_axes {
        let axis = scirs2_core::ndarray_ext::Axis(axis_idx);
        result = match op {
            ReduceOp::Sum => result.sum_axis(axis),
            ReduceOp::Max => result.map_axis(axis, |view| {
                view.iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(T::default)
            }),
            ReduceOp::Min => result.map_axis(axis, |view| {
                view.iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(T::default)
            }),
            ReduceOp::Mean => result.mean_axis(axis).ok_or_else(|| {
                anyhow!(
                    "Mean reduction failed - axis might be empty or type doesn't support division"
                )
            })?,
            ReduceOp::Prod => result.map_axis(axis, |view| {
                view.iter().cloned().fold(T::one(), |acc, x| acc * x)
            }),
            ReduceOp::All => result.map_axis(axis, |view| {
                let all_nonzero = view.iter().all(|&x| x != T::zero());
                if all_nonzero {
                    T::one()
                } else {
                    T::zero()
                }
            }),
            ReduceOp::Any => result.map_axis(axis, |view| {
                let any_nonzero = view.iter().any(|&x| x != T::zero());
                if any_nonzero {
                    T::one()
                } else {
                    T::zero()
                }
            }),
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                return Err(anyhow!(
                    "ArgMax and ArgMin should use dedicated argmax/argmin methods, not reduce"
                ));
            }
        };
    }
    let result_tensor = DenseND::from_array(result);
    Ok(TensorHandle::from_dense_auto(result_tensor))
}

pub(super) fn softmax<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for softmax"))?;
    let ndim = dense.shape().len();
    if axis >= ndim {
        return Err(anyhow!(
            "Axis {} out of range for tensor with {} dimensions",
            axis,
            ndim
        ));
    }
    use scirs2_core::ndarray_ext::Zip;
    let axis_obj = scirs2_core::ndarray_ext::Axis(axis);
    let max_vals = dense.view().map_axis(axis_obj, |view| {
        view.iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(T::zero)
    });
    let mut exp_vals = dense.view().to_owned();
    Zip::from(exp_vals.lanes_mut(axis_obj))
        .and(max_vals.view())
        .for_each(|mut lane, &max_val| {
            lane.mapv_inplace(|v| (v - max_val).exp());
        });
    let sum_exp = exp_vals.sum_axis(axis_obj);
    let mut result = exp_vals;
    Zip::from(result.lanes_mut(axis_obj))
        .and(sum_exp.view())
        .for_each(|mut lane, &sum_val| {
            lane.mapv_inplace(|v| v / sum_val);
        });
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(result)))
}

pub(super) fn log_softmax<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for log_softmax"))?;
    let ndim = dense.shape().len();
    if axis >= ndim {
        return Err(anyhow!(
            "Axis {} out of range for tensor with {} dimensions",
            axis,
            ndim
        ));
    }
    use scirs2_core::ndarray_ext::Zip;
    let axis_obj = scirs2_core::ndarray_ext::Axis(axis);
    let max_vals = dense.view().map_axis(axis_obj, |view| {
        view.iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(T::zero)
    });
    let mut exp_vals = dense.view().to_owned();
    Zip::from(exp_vals.lanes_mut(axis_obj))
        .and(max_vals.view())
        .for_each(|mut lane, &max_val| {
            lane.mapv_inplace(|v| (v - max_val).exp());
        });
    let sum_exp = exp_vals.sum_axis(axis_obj);
    let log_sum_exp = sum_exp.mapv(|v| v.ln());
    let mut result = dense.view().to_owned();
    Zip::from(result.lanes_mut(axis_obj))
        .and(max_vals.view())
        .and(log_sum_exp.view())
        .for_each(|mut lane, &max_val, &lse| {
            lane.mapv_inplace(|v| v - max_val - lse);
        });
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(result)))
}

pub(super) fn layer_norm<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    eps: T,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for layer_norm"))?;
    let ndim = dense.shape().len();
    if ndim == 0 {
        return Err(anyhow!("Cannot normalize scalar tensor"));
    }
    let last_axis = ndim - 1;
    use scirs2_core::ndarray_ext::{Axis as NdAxis, Zip};
    let mean = dense.view().mean_axis(NdAxis(last_axis)).ok_or_else(|| {
        anyhow!("Mean computation failed - axis might be empty or type doesn't support division")
    })?;
    let mut variance = dense.view().to_owned();
    Zip::from(variance.lanes_mut(NdAxis(last_axis)))
        .and(mean.view())
        .for_each(|mut lane, &m| {
            lane.mapv_inplace(|v| {
                let diff = v - m;
                diff * diff
            });
        });
    let variance = variance
        .mean_axis(NdAxis(last_axis))
        .ok_or_else(|| anyhow!("Variance computation failed"))?;
    let mut result = dense.view().to_owned();
    Zip::from(result.lanes_mut(NdAxis(last_axis)))
        .and(mean.view())
        .and(variance.view())
        .for_each(|mut lane, &m, &v| {
            let std = (v + eps).sqrt();
            lane.mapv_inplace(|x_val| (x_val - m) / std);
        });
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(result)))
}

pub(super) fn batch_norm<T>(
    _executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    eps: T,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for batch_norm"))?;
    let ndim = dense.shape().len();
    if ndim == 0 {
        return Err(anyhow!("Cannot normalize scalar tensor"));
    }
    let batch_axis = 0;
    use scirs2_core::ndarray_ext::{Axis as NdAxis, Zip};
    let mean = dense.view().mean_axis(NdAxis(batch_axis)).ok_or_else(|| {
        anyhow!("Mean computation failed - axis might be empty or type doesn't support division")
    })?;
    let mut variance = dense.view().to_owned();
    Zip::from(variance.lanes_mut(NdAxis(batch_axis)))
        .and(mean.view())
        .for_each(|mut lane, &m| {
            lane.mapv_inplace(|v| {
                let diff = v - m;
                diff * diff
            });
        });
    let variance = variance
        .mean_axis(NdAxis(batch_axis))
        .ok_or_else(|| anyhow!("Variance computation failed"))?;
    let mut result = dense.view().to_owned();
    Zip::from(result.lanes_mut(NdAxis(batch_axis)))
        .and(mean.view())
        .and(variance.view())
        .for_each(|mut lane, &m, &v| {
            let std = (v + eps).sqrt();
            lane.mapv_inplace(|x_val| (x_val - m) / std);
        });
    Ok(TensorHandle::from_dense_auto(DenseND::from_array(result)))
}

pub(super) fn argmax<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for argmax"))?;

    let shape = dense.shape();

    if axis >= shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        ));
    }

    // Calculate output shape (remove the reduction axis)
    let output_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();

    let input_view = dense.view();
    let output_size: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };
    let mut output_data = Vec::with_capacity(output_size);

    // For each output position, find the argmax along the reduction axis
    for i in 0..output_size {
        let base_idx = if output_shape.is_empty() {
            vec![]
        } else {
            executor.flat_to_multidim(i, &output_shape)
        };

        let mut max_val = T::from_f64(f64::NEG_INFINITY)
            .ok_or_else(|| anyhow!("Failed to convert NEG_INFINITY to generic float type"))?;
        let mut max_idx = 0usize;

        for j in 0..shape[axis] {
            let mut idx = Vec::new();
            let mut out_pos = 0;
            for (dim, &_size) in shape.iter().enumerate() {
                if dim == axis {
                    idx.push(j);
                } else {
                    idx.push(base_idx[out_pos]);
                    out_pos += 1;
                }
            }
            let val = input_view[idx.as_slice()];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }

        output_data.push(
            T::from_usize(max_idx)
                .ok_or_else(|| anyhow!("Failed to convert index to generic float type"))?,
        );
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_shape = if output_shape.is_empty() {
        vec![]
    } else {
        output_shape
    };
    let result_array = Array::from_shape_vec(IxDyn(&result_shape), output_data)
        .map_err(|e| anyhow!("Failed to create argmax array: {}", e))?;

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}

pub(super) fn argmin<T>(
    executor: &mut CpuExecutor,
    x: &TensorHandle<T>,
    axis: Axis,
) -> Result<TensorHandle<T>>
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    let dense = x
        .as_dense()
        .ok_or_else(|| anyhow!("Only dense tensors supported for argmin"))?;

    let shape = dense.shape();

    if axis >= shape.len() {
        return Err(anyhow!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        ));
    }

    // Calculate output shape (remove the reduction axis)
    let output_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();

    let input_view = dense.view();
    let output_size: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };
    let mut output_data = Vec::with_capacity(output_size);

    // For each output position, find the argmin along the reduction axis
    for i in 0..output_size {
        let base_idx = if output_shape.is_empty() {
            vec![]
        } else {
            executor.flat_to_multidim(i, &output_shape)
        };

        let mut min_val = T::from_f64(f64::INFINITY)
            .ok_or_else(|| anyhow!("Failed to convert INFINITY to generic float type"))?;
        let mut min_idx = 0usize;

        for j in 0..shape[axis] {
            let mut idx = Vec::new();
            let mut out_pos = 0;
            for (dim, &_size) in shape.iter().enumerate() {
                if dim == axis {
                    idx.push(j);
                } else {
                    idx.push(base_idx[out_pos]);
                    out_pos += 1;
                }
            }
            let val = input_view[idx.as_slice()];
            if val < min_val {
                min_val = val;
                min_idx = j;
            }
        }

        output_data.push(
            T::from_usize(min_idx)
                .ok_or_else(|| anyhow!("Failed to convert index to generic float type"))?,
        );
    }

    use scirs2_core::ndarray_ext::{Array, IxDyn};
    let result_shape = if output_shape.is_empty() {
        vec![]
    } else {
        output_shape
    };
    let result_array = Array::from_shape_vec(IxDyn(&result_shape), output_data)
        .map_err(|e| anyhow!("Failed to create argmin array: {}", e))?;

    Ok(TensorHandle::from_dense_auto(DenseND::from_array(
        result_array,
    )))
}
