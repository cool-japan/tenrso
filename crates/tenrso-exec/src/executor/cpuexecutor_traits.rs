//! # CpuExecutor - Trait Implementations
//!
//! This module contains trait implementations for `CpuExecutor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `TenrsoExecutor`
//!
//! ## File Size Note
//!
//! This file is 2,152 lines (7.6% over the 2000-line policy limit).
//! It contains a single, coherent `impl TenrsoExecutor<T> for CpuExecutor` block
//! with 42 trait methods. Splitting this impl block across multiple files would
//! require significant architectural changes (delegation pattern, helper traits, etc.)
//! and would compromise code cohesion. The methods are already organized by category
//! and delegate to specialized modules where appropriate (simd_ops, tiled_reductions,
//! advanced_indexing, etc.).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use crate::hints::ExecHints;
use anyhow::{anyhow, Result};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND, TensorHandle};
use tenrso_planner::EinsumSpec;

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TenrsoExecutor<T> for CpuExecutor
where
    T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive + 'static,
{
    fn einsum(
        &mut self,
        spec: &str,
        inputs: &[TensorHandle<T>],
        hints: &ExecHints,
    ) -> Result<TensorHandle<T>> {
        let parsed_spec = EinsumSpec::parse(spec)?;
        if parsed_spec.num_inputs() != inputs.len() {
            return Err(anyhow!(
                "Spec expects {} inputs, got {}",
                parsed_spec.num_inputs(),
                inputs.len()
            ));
        }
        let dense_inputs: Vec<&DenseND<T>> = inputs
            .iter()
            .map(|h| {
                h.as_dense()
                    .ok_or_else(|| anyhow!("Only dense tensors supported for now"))
            })
            .collect::<Result<Vec<_>>>()?;
        let dense_inputs_owned: Vec<DenseND<T>> = dense_inputs.iter().map(|&t| t.clone()).collect();
        let result = self.execute_einsum_with_planner(&parsed_spec, &dense_inputs_owned, hints)?;
        Ok(TensorHandle::from_dense_auto(result))
    }
    fn elem_op(&mut self, op: ElemOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for elem_op"))?;
        let result_data = match op {
            ElemOp::Neg => dense.view().mapv(|v| -v),
            ElemOp::Abs => dense.view().mapv(|v| v.abs()),
            ElemOp::Exp => dense.view().mapv(|v| v.exp()),
            ElemOp::Log => dense.view().mapv(|v| v.ln()),
            ElemOp::Sin => dense.view().mapv(|v| v.sin()),
            ElemOp::Cos => dense.view().mapv(|v| v.cos()),
            ElemOp::Sqrt => dense.view().mapv(|v| v.sqrt()),
            ElemOp::Sqr => dense.view().mapv(|v| v * v),
            ElemOp::Recip => dense.view().mapv(|v| v.recip()),
            ElemOp::Tanh => dense.view().mapv(|v| v.tanh()),
            ElemOp::Sigmoid => dense.view().mapv(|v| {
                let one = T::one();
                one / (one + (-v).exp())
            }),
            ElemOp::ReLU => dense.view().mapv(|v| {
                let zero = T::zero();
                if v > zero {
                    v
                } else {
                    zero
                }
            }),
            ElemOp::Gelu => dense.view().mapv(|v| {
                let half = T::from_f64(0.5).unwrap_or_else(T::one);
                let one = T::one();
                let coeff = T::from_f64(0.7978845608028654).unwrap_or_else(T::one);
                let cubic_coeff = T::from_f64(0.044715).unwrap_or_else(T::zero);
                let x_cubed = v * v * v;
                let inner = coeff * (v + cubic_coeff * x_cubed);
                half * v * (one + inner.tanh())
            }),
            ElemOp::Elu => dense.view().mapv(|v| {
                let zero = T::zero();
                let one = T::one();
                if v > zero {
                    v
                } else {
                    v.exp() - one
                }
            }),
            ElemOp::Selu => dense.view().mapv(|v| {
                let zero = T::zero();
                let one = T::one();
                let scale = T::from_f64(1.050_700_987_355_480_5).unwrap_or_else(T::one);
                let alpha = T::from_f64(1.673_263_242_354_377_2).unwrap_or_else(T::one);
                if v > zero {
                    scale * v
                } else {
                    scale * alpha * (v.exp() - one)
                }
            }),
            ElemOp::Softplus => dense.view().mapv(|v| {
                let zero = T::zero();
                let one = T::one();
                let abs_v = v.abs();
                let max_part = if v > zero { v } else { zero };
                max_part + (one + (-abs_v).exp()).ln()
            }),
            ElemOp::Sign => dense.view().mapv(|v| {
                let zero = T::zero();
                let one = T::one();
                let neg_one = -one;
                if v > zero {
                    one
                } else if v < zero {
                    neg_one
                } else {
                    zero
                }
            }),
        };
        let result = DenseND::from_array(result_data);
        Ok(TensorHandle::from_dense_auto(result))
    }
    fn binary_op(
        &mut self,
        op: BinaryOp,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        let dense_x = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for binary_op"))?;
        let dense_y = y
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for binary_op"))?;
        if dense_x.shape() != dense_y.shape() {
            return self.binary_op_with_broadcast(op, dense_x, dense_y);
        }
        use scirs2_core::ndarray_ext::Zip;
        let result_data = match op {
            BinaryOp::Add => &dense_x.view() + &dense_y.view(),
            BinaryOp::Sub => &dense_x.view() - &dense_y.view(),
            BinaryOp::Mul => &dense_x.view() * &dense_y.view(),
            BinaryOp::Div => &dense_x.view() / &dense_y.view(),
            BinaryOp::Pow => {
                let mut result = dense_x.view().to_owned();
                Zip::from(&mut result)
                    .and(&dense_x.view())
                    .and(&dense_y.view())
                    .for_each(|r, &x_val, &y_val| {
                        *r = x_val.powf(y_val);
                    });
                result
            }
            BinaryOp::Maximum => {
                let mut result = dense_x.view().to_owned();
                Zip::from(&mut result)
                    .and(&dense_x.view())
                    .and(&dense_y.view())
                    .for_each(|r, &x_val, &y_val| {
                        *r = if x_val > y_val { x_val } else { y_val };
                    });
                result
            }
            BinaryOp::Minimum => {
                let mut result = dense_x.view().to_owned();
                Zip::from(&mut result)
                    .and(&dense_x.view())
                    .and(&dense_y.view())
                    .for_each(|r, &x_val, &y_val| {
                        *r = if x_val < y_val { x_val } else { y_val };
                    });
                result
            }
        };
        let result = DenseND::from_array(result_data);
        Ok(TensorHandle::from_dense_auto(result))
    }
    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &TensorHandle<T>,
        axes: &[Axis],
    ) -> Result<TensorHandle<T>> {
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
                ReduceOp::Max => {
                    result
                        .map_axis(
                            axis,
                            |view| {
                                view.iter()
                                    .cloned()
                                    .max_by(|a, b| {
                                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .unwrap_or_else(T::default)
                            },
                        )
                }
                ReduceOp::Min => {
                    result
                        .map_axis(
                            axis,
                            |view| {
                                view.iter()
                                    .cloned()
                                    .min_by(|a, b| {
                                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .unwrap_or_else(T::default)
                            },
                        )
                }
                ReduceOp::Mean => {
                    result
                        .mean_axis(axis)
                        .ok_or_else(|| {
                            anyhow!(
                                "Mean reduction failed - axis might be empty or type doesn't support division"
                            )
                        })?
                }
                ReduceOp::Prod => {
                    result
                        .map_axis(
                            axis,
                            |view| {
                                view.iter()
                                    .cloned()
                                    .fold(T::one(), |acc, x| acc * x)
                            },
                        )
                }
                ReduceOp::All => {
                    result
                        .map_axis(
                            axis,
                            |view| {
                                let all_nonzero = view.iter().all(|&x| x != T::zero());
                                if all_nonzero { T::one() } else { T::zero() }
                            },
                        )
                }
                ReduceOp::Any => {
                    result
                        .map_axis(
                            axis,
                            |view| {
                                let any_nonzero = view.iter().any(|&x| x != T::zero());
                                if any_nonzero { T::one() } else { T::zero() }
                            },
                        )
                }
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
    fn clip(&mut self, x: &TensorHandle<T>, min_val: T, max_val: T) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for clip"))?;
        if min_val > max_val {
            return Err(anyhow!("Invalid clip bounds: min_val > max_val"));
        }
        let result_data = dense.view().mapv(|v| {
            if v < min_val {
                min_val
            } else if v > max_val {
                max_val
            } else {
                v
            }
        });
        let result = DenseND::from_array(result_data);
        Ok(TensorHandle::from_dense_auto(result))
    }
    fn softmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
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
    fn log_softmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
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
    fn transpose(&mut self, x: &TensorHandle<T>, axes: &[Axis]) -> Result<TensorHandle<T>> {
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
    fn reshape(&mut self, x: &TensorHandle<T>, new_shape: &[usize]) -> Result<TensorHandle<T>> {
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
    fn concatenate(&mut self, tensors: &[TensorHandle<T>], axis: Axis) -> Result<TensorHandle<T>> {
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
        let mut output_data = self.acquire_pooled_generic::<T>(&output_shape);
        output_data.clear(); // Ensure buffer starts empty
        output_data.reserve(output_size);

        for flat_idx in 0..output_size {
            let out_idx = self.flat_to_multidim(flat_idx, &output_shape);
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
        self.release_pooled_generic::<T>(&output_shape, output_data);
        let result = DenseND::from_array(result_array);
        Ok(TensorHandle::from_dense_auto(result))
    }
    fn split(
        &mut self,
        x: &TensorHandle<T>,
        num_splits: usize,
        axis: Axis,
    ) -> Result<Vec<TensorHandle<T>>> {
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
    fn layer_norm(&mut self, x: &TensorHandle<T>, eps: T) -> Result<TensorHandle<T>> {
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
            anyhow!(
                "Mean computation failed - axis might be empty or type doesn't support division"
            )
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
    fn batch_norm(&mut self, x: &TensorHandle<T>, eps: T) -> Result<TensorHandle<T>> {
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
            anyhow!(
                "Mean computation failed - axis might be empty or type doesn't support division"
            )
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
    fn where_op(
        &mut self,
        condition: &TensorHandle<T>,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
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
    fn masked_select(
        &mut self,
        x: &TensorHandle<T>,
        mask: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
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
    fn modulo(&mut self, x: &TensorHandle<T>, divisor: T) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for modulo"))?;
        if divisor == T::zero() {
            return Err(anyhow!("Division by zero in modulo operation"));
        }
        let result_data = dense.view().mapv(|v| {
            let quot = (v / divisor).floor();
            v - quot * divisor
        });
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_data,
        )))
    }
    fn remainder(&mut self, x: &TensorHandle<T>, divisor: T) -> Result<TensorHandle<T>> {
        self.modulo(x, divisor)
    }
    fn max_pool_1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for max_pool_1d"))?;
        if kernel_size == 0 || stride == 0 {
            return Err(anyhow!("Kernel size and stride must be positive"));
        }
        let shape = dense.shape();
        if shape.len() != 1 {
            return Err(anyhow!(
                "Expected 1D tensor for max_pool_1d, got {:?}D",
                shape.len()
            ));
        }
        let input_len = shape[0];
        if kernel_size > input_len {
            return Err(anyhow!(
                "Kernel size {} larger than input length {}",
                kernel_size,
                input_len
            ));
        }
        let output_len = (input_len - kernel_size) / stride + 1;
        let mut output = Vec::with_capacity(output_len);
        let view = dense.view();
        for i in 0..output_len {
            let start = i * stride;
            let end = start + kernel_size;
            let max_val = (start..end)
                .map(|j| view[[j]])
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(T::default);
            output.push(max_val);
        }
        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&[output_len]), output)
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn avg_pool_1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: usize,
        stride: usize,
    ) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for avg_pool_1d"))?;
        if kernel_size == 0 || stride == 0 {
            return Err(anyhow!("Kernel size and stride must be positive"));
        }
        let shape = dense.shape();
        if shape.len() != 1 {
            return Err(anyhow!(
                "Expected 1D tensor for avg_pool_1d, got {:?}D",
                shape.len()
            ));
        }
        let input_len = shape[0];
        if kernel_size > input_len {
            return Err(anyhow!(
                "Kernel size {} larger than input length {}",
                kernel_size,
                input_len
            ));
        }
        let output_len = (input_len - kernel_size) / stride + 1;
        let mut output = Vec::with_capacity(output_len);
        let view = dense.view();
        let kernel_size_t = T::from_usize(kernel_size).unwrap_or_else(T::one);
        for i in 0..output_len {
            let start = i * stride;
            let end = start + kernel_size;
            let mut sum = T::zero();
            for j in start..end {
                sum += view[[j]];
            }
            let avg = sum / kernel_size_t;
            output.push(avg);
        }
        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&[output_len]), output)
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn max_pool_2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for max_pool_2d"))?;
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(anyhow!("Kernel size and stride must be positive"));
        }
        let shape = dense.shape();
        if shape.len() != 2 {
            return Err(anyhow!(
                "Expected 2D tensor for max_pool_2d, got {:?}D",
                shape.len()
            ));
        }
        let (h, w) = (shape[0], shape[1]);
        if kh > h || kw > w {
            return Err(anyhow!(
                "Kernel size ({}, {}) larger than input ({}, {})",
                kh,
                kw,
                h,
                w
            ));
        }
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let output_shape = [out_h, out_w];

        // Use pooled buffer for max_pool_2d output (Phase 5: Automatic Pooling)
        let mut output = self.acquire_pooled_generic::<T>(&output_shape);
        output.clear(); // Ensure buffer starts empty
        output.reserve(out_h * out_w);

        let view = dense.view();
        for i in 0..out_h {
            for j in 0..out_w {
                let start_h = i * sh;
                let start_w = j * sw;
                let mut max_val = T::default();
                let mut first = true;
                for di in 0..kh {
                    for dj in 0..kw {
                        let val = view[[start_h + di, start_w + dj]];
                        if first || val > max_val {
                            max_val = val;
                            first = false;
                        }
                    }
                }
                output.push(max_val);
            }
        }

        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn avg_pool_2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for avg_pool_2d"))?;
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(anyhow!("Kernel size and stride must be positive"));
        }
        let shape = dense.shape();
        if shape.len() != 2 {
            return Err(anyhow!(
                "Expected 2D tensor for avg_pool_2d, got {:?}D",
                shape.len()
            ));
        }
        let (h, w) = (shape[0], shape[1]);
        if kh > h || kw > w {
            return Err(anyhow!(
                "Kernel size ({}, {}) larger than input ({}, {})",
                kh,
                kw,
                h,
                w
            ));
        }
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let output_shape = [out_h, out_w];

        // Use pooled buffer for avg_pool_2d output (Phase 5: Automatic Pooling)
        let mut output = self.acquire_pooled_generic::<T>(&output_shape);
        output.clear(); // Ensure buffer starts empty
        output.reserve(out_h * out_w);

        let view = dense.view();
        let kernel_count = T::from_usize(kh * kw).unwrap_or_else(T::one);
        for i in 0..out_h {
            for j in 0..out_w {
                let start_h = i * sh;
                let start_w = j * sw;
                let mut sum = T::zero();
                for di in 0..kh {
                    for dj in 0..kw {
                        sum += view[[start_h + di, start_w + dj]];
                    }
                }
                let avg = sum / kernel_count;
                output.push(avg);
            }
        }

        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn conv1d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: usize,
        padding: (usize, usize),
    ) -> Result<TensorHandle<T>> {
        let dense_x = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for conv1d"))?;
        let dense_kernel = kernel
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for conv1d kernel"))?;
        if stride == 0 {
            return Err(anyhow!("Stride must be positive"));
        }
        let x_shape = dense_x.shape();
        let k_shape = dense_kernel.shape();
        if x_shape.len() != 3 {
            return Err(anyhow!(
                "Expected 3D input tensor [batch, in_channels, length], got {:?}D",
                x_shape.len()
            ));
        }
        if k_shape.len() != 3 {
            return Err(anyhow!(
                "Expected 3D kernel tensor [out_channels, in_channels, kernel_size], got {:?}D",
                k_shape.len()
            ));
        }
        let (batch, in_channels, in_length) = (x_shape[0], x_shape[1], x_shape[2]);
        let (out_channels, k_in_channels, kernel_size) = (k_shape[0], k_shape[1], k_shape[2]);
        if in_channels != k_in_channels {
            return Err(anyhow!(
                "Input channels mismatch: input has {}, kernel expects {}",
                in_channels,
                k_in_channels
            ));
        }
        if let Some(bias_tensor) = bias {
            let bias_dense = bias_tensor
                .as_dense()
                .ok_or_else(|| anyhow!("Only dense tensors supported for bias"))?;
            let bias_shape = bias_dense.shape();
            if bias_shape.len() != 1 || bias_shape[0] != out_channels {
                return Err(anyhow!(
                    "Expected bias shape [{}], got {:?}",
                    out_channels,
                    bias_shape
                ));
            }
        }
        let (pad_left, pad_right) = padding;
        let padded_length = in_length + pad_left + pad_right;
        if kernel_size > padded_length {
            return Err(anyhow!(
                "Kernel size {} larger than padded input length {}",
                kernel_size,
                padded_length
            ));
        }
        let out_length = (padded_length - kernel_size) / stride + 1;
        let output_shape = [batch, out_channels, out_length];

        // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
        let mut output = self.acquire_pooled_generic::<T>(&output_shape);
        output.clear(); // Ensure buffer starts empty
        output.resize(batch * out_channels * out_length, T::zero());

        let x_view = dense_x.view();
        let k_view = dense_kernel.view();
        for b in 0..batch {
            for oc in 0..out_channels {
                for o in 0..out_length {
                    let mut sum = T::zero();
                    let in_start = (o * stride) as isize - pad_left as isize;
                    for ic in 0..in_channels {
                        for k in 0..kernel_size {
                            let in_pos = in_start + k as isize;
                            if in_pos >= 0 && (in_pos as usize) < in_length {
                                let x_val = x_view[[b, ic, in_pos as usize]];
                                let k_val = k_view[[oc, ic, k]];
                                sum += x_val * k_val;
                            }
                        }
                    }
                    if let Some(bias_tensor) = bias {
                        let bias_dense = bias_tensor.as_dense().unwrap();
                        let bias_view = bias_dense.view();
                        sum += bias_view[[oc]];
                    }
                    output[b * out_channels * out_length + oc * out_length + o] = sum;
                }
            }
        }

        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn conv2d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
    ) -> Result<TensorHandle<T>> {
        let dense_x = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for conv2d"))?;
        let dense_kernel = kernel
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for conv2d kernel"))?;
        let (stride_h, stride_w) = stride;
        if stride_h == 0 || stride_w == 0 {
            return Err(anyhow!("Stride must be positive"));
        }
        let x_shape = dense_x.shape();
        let k_shape = dense_kernel.shape();
        if x_shape.len() != 4 {
            return Err(anyhow!(
                "Expected 4D input tensor [batch, in_channels, height, width], got {:?}D",
                x_shape.len()
            ));
        }
        if k_shape.len() != 4 {
            return Err(
                anyhow!(
                    "Expected 4D kernel tensor [out_channels, in_channels, kernel_h, kernel_w], got {:?}D",
                    k_shape.len()
                ),
            );
        }
        let (batch, in_channels, in_h, in_w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (out_channels, k_in_channels, kernel_h, kernel_w) =
            (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
        if in_channels != k_in_channels {
            return Err(anyhow!(
                "Input channels mismatch: input has {}, kernel expects {}",
                in_channels,
                k_in_channels
            ));
        }
        if let Some(bias_tensor) = bias {
            let bias_dense = bias_tensor
                .as_dense()
                .ok_or_else(|| anyhow!("Only dense tensors supported for bias"))?;
            let bias_shape = bias_dense.shape();
            if bias_shape.len() != 1 || bias_shape[0] != out_channels {
                return Err(anyhow!(
                    "Expected bias shape [{}], got {:?}",
                    out_channels,
                    bias_shape
                ));
            }
        }
        let (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right) = padding;
        let padded_h = in_h + pad_h_top + pad_h_bottom;
        let padded_w = in_w + pad_w_left + pad_w_right;
        if kernel_h > padded_h || kernel_w > padded_w {
            return Err(anyhow!(
                "Kernel size ({}, {}) larger than padded input ({}, {})",
                kernel_h,
                kernel_w,
                padded_h,
                padded_w
            ));
        }
        let out_h = (padded_h - kernel_h) / stride_h + 1;
        let out_w = (padded_w - kernel_w) / stride_w + 1;
        let output_shape = [batch, out_channels, out_h, out_w];

        // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
        let mut output = self.acquire_pooled_generic::<T>(&output_shape);
        output.clear(); // Ensure buffer starts empty
        output.resize(batch * out_channels * out_h * out_w, T::zero());

        let x_view = dense_x.view();
        let k_view = dense_kernel.view();
        for b in 0..batch {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = T::zero();
                        let in_start_h = (oh * stride_h) as isize - pad_h_top as isize;
                        let in_start_w = (ow * stride_w) as isize - pad_w_left as isize;
                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let in_h_pos = in_start_h + kh as isize;
                                    let in_w_pos = in_start_w + kw as isize;
                                    if in_h_pos >= 0
                                        && (in_h_pos as usize) < in_h
                                        && in_w_pos >= 0
                                        && (in_w_pos as usize) < in_w
                                    {
                                        let x_val =
                                            x_view[[b, ic, in_h_pos as usize, in_w_pos as usize]];
                                        let k_val = k_view[[oc, ic, kh, kw]];
                                        sum += x_val * k_val;
                                    }
                                }
                            }
                        }
                        if let Some(bias_tensor) = bias {
                            let bias_dense = bias_tensor.as_dense().unwrap();
                            let bias_view = bias_dense.view();
                            sum += bias_view[[oc]];
                        }
                        let out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn gather(
        &mut self,
        x: &TensorHandle<T>,
        axis: Axis,
        indices: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
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
                let idx_multi = self.flat_to_multidim(idx_flat, indices_shape);
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
                    let elem_multi = self.flat_to_multidim(elem_idx, &x_shape[1..]);
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
    fn scatter(
        &mut self,
        shape: &[usize],
        axis: Axis,
        indices: &TensorHandle<T>,
        values: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
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
                let idx_multi = self.flat_to_multidim(idx_flat, indices_shape);
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
                    let elem_multi = self.flat_to_multidim(elem_idx, &shape[1..]);
                    values_index.extend(elem_multi.clone());
                    let mut out_index = vec![idx];
                    out_index.extend(elem_multi);
                    let out_flat = self.multidim_to_flat(&out_index, shape);
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
    fn conv3d(
        &mut self,
        x: &TensorHandle<T>,
        kernel: &TensorHandle<T>,
        bias: Option<&TensorHandle<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize, usize, usize, usize),
    ) -> Result<TensorHandle<T>> {
        let dense_x = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for conv3d"))?;
        let dense_kernel = kernel
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for conv3d kernel"))?;
        let (stride_d, stride_h, stride_w) = stride;
        if stride_d == 0 || stride_h == 0 || stride_w == 0 {
            return Err(anyhow!("Stride must be positive"));
        }
        let x_shape = dense_x.shape();
        let k_shape = dense_kernel.shape();
        if x_shape.len() != 5 {
            return Err(anyhow!(
                "Expected 5D input tensor [batch, in_channels, depth, height, width], got {:?}D",
                x_shape.len()
            ));
        }
        if k_shape.len() != 5 {
            return Err(
                anyhow!(
                    "Expected 5D kernel tensor [out_channels, in_channels, kernel_d, kernel_h, kernel_w], got {:?}D",
                    k_shape.len()
                ),
            );
        }
        let (batch, in_channels, in_d, in_h, in_w) =
            (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]);
        let (out_channels, k_in_channels, kernel_d, kernel_h, kernel_w) =
            (k_shape[0], k_shape[1], k_shape[2], k_shape[3], k_shape[4]);
        if in_channels != k_in_channels {
            return Err(anyhow!(
                "Input channels mismatch: input has {}, kernel expects {}",
                in_channels,
                k_in_channels
            ));
        }
        if let Some(bias_tensor) = bias {
            let bias_dense = bias_tensor
                .as_dense()
                .ok_or_else(|| anyhow!("Only dense tensors supported for bias"))?;
            let bias_shape = bias_dense.shape();
            if bias_shape.len() != 1 || bias_shape[0] != out_channels {
                return Err(anyhow!(
                    "Expected bias shape [{}], got {:?}",
                    out_channels,
                    bias_shape
                ));
            }
        }
        let (pad_d_front, pad_d_back, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right) = padding;
        let padded_d = in_d + pad_d_front + pad_d_back;
        let padded_h = in_h + pad_h_top + pad_h_bottom;
        let padded_w = in_w + pad_w_left + pad_w_right;
        if kernel_d > padded_d || kernel_h > padded_h || kernel_w > padded_w {
            return Err(anyhow!(
                "Kernel size ({}, {}, {}) larger than padded input ({}, {}, {})",
                kernel_d,
                kernel_h,
                kernel_w,
                padded_d,
                padded_h,
                padded_w
            ));
        }
        let out_d = (padded_d - kernel_d) / stride_d + 1;
        let out_h = (padded_h - kernel_h) / stride_h + 1;
        let out_w = (padded_w - kernel_w) / stride_w + 1;
        let output_shape = [batch, out_channels, out_d, out_h, out_w];

        // Use pooled buffer for output allocation (Phase 5: Automatic Pooling)
        let mut output = self.acquire_pooled_generic::<T>(&output_shape);
        output.clear(); // Ensure buffer starts empty
        output.resize(batch * out_channels * out_d * out_h * out_w, T::zero());

        let x_view = dense_x.view();
        let k_view = dense_kernel.view();
        for b in 0..batch {
            for oc in 0..out_channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = T::zero();
                            let in_start_d = (od * stride_d) as isize - pad_d_front as isize;
                            let in_start_h = (oh * stride_h) as isize - pad_h_top as isize;
                            let in_start_w = (ow * stride_w) as isize - pad_w_left as isize;
                            for ic in 0..in_channels {
                                for kd in 0..kernel_d {
                                    for kh in 0..kernel_h {
                                        for kw in 0..kernel_w {
                                            let in_d_pos = in_start_d + kd as isize;
                                            let in_h_pos = in_start_h + kh as isize;
                                            let in_w_pos = in_start_w + kw as isize;
                                            if in_d_pos >= 0
                                                && (in_d_pos as usize) < in_d
                                                && in_h_pos >= 0
                                                && (in_h_pos as usize) < in_h
                                                && in_w_pos >= 0
                                                && (in_w_pos as usize) < in_w
                                            {
                                                let x_val = x_view[[
                                                    b,
                                                    ic,
                                                    in_d_pos as usize,
                                                    in_h_pos as usize,
                                                    in_w_pos as usize,
                                                ]];
                                                let k_val = k_view[[oc, ic, kd, kh, kw]];
                                                sum += x_val * k_val;
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(bias_tensor) = bias {
                                let bias_dense = bias_tensor.as_dense().unwrap();
                                let bias_view = bias_dense.view();
                                sum += bias_view[[oc]];
                            }
                            let out_idx = ((((b * out_channels + oc) * out_d + od) * out_h + oh)
                                * out_w)
                                + ow;
                            output[out_idx] = sum;
                        }
                    }
                }
            }
        }

        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output.clone())
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn determinant(&mut self, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for determinant"))?;
        let shape = dense.shape();
        if shape.len() < 2 {
            return Err(anyhow!("Input must be at least 2D for determinant"));
        }
        let n = shape[shape.len() - 1];
        let m = shape[shape.len() - 2];
        if n != m {
            return Err(anyhow!(
                "Last two dimensions must be square for determinant, got {}x{}",
                m,
                n
            ));
        }
        if shape.len() == 2 {
            use scirs2_core::ndarray_ext::Array2;
            let view = dense.view();
            let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| view[[i, j]]);
            let det = self.compute_determinant_2d(&matrix)?;
            return Ok(TensorHandle::from_dense_auto(DenseND::from_vec(
                vec![det],
                &[],
            )?));
        }
        let batch_size: usize = shape[..shape.len() - 2].iter().product();
        let mut determinants = Vec::with_capacity(batch_size);
        let view = dense.view();
        for batch_idx in 0..batch_size {
            let batch_multi = self.flat_to_multidim(batch_idx, &shape[..shape.len() - 2]);
            use scirs2_core::ndarray_ext::Array2;
            let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| {
                let mut idx = batch_multi.clone();
                idx.push(i);
                idx.push(j);
                view[idx.as_slice()]
            });
            let det = self.compute_determinant_2d(&matrix)?;
            determinants.push(det);
        }
        let output_shape = &shape[..shape.len() - 2];
        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(output_shape), determinants)
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn matrix_inverse(&mut self, x: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for matrix_inverse"))?;
        let shape = dense.shape();
        if shape.len() < 2 {
            return Err(anyhow!("Input must be at least 2D for matrix inverse"));
        }
        let n = shape[shape.len() - 1];
        let m = shape[shape.len() - 2];
        if n != m {
            return Err(anyhow!(
                "Last two dimensions must be square for matrix inverse, got {}x{}",
                m,
                n
            ));
        }
        if shape.len() == 2 {
            use scirs2_core::ndarray_ext::Array2;
            let view = dense.view();
            let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| view[[i, j]]);
            let inv = self.compute_inverse_2d(&matrix)?;
            let inv_dyn = inv.into_dyn();
            return Ok(TensorHandle::from_dense_auto(DenseND::from_array(inv_dyn)));
        }
        let batch_size: usize = shape[..shape.len() - 2].iter().product();
        let output_size = batch_size * n * n;
        let mut output = Vec::with_capacity(output_size);
        let view = dense.view();
        for batch_idx in 0..batch_size {
            let batch_multi = self.flat_to_multidim(batch_idx, &shape[..shape.len() - 2]);
            use scirs2_core::ndarray_ext::Array2;
            let matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| {
                let mut idx = batch_multi.clone();
                idx.push(i);
                idx.push(j);
                view[idx.as_slice()]
            });
            let inv = self.compute_inverse_2d(&matrix)?;
            output.extend(inv.iter().copied());
        }
        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(shape), output)
            .map_err(|e| anyhow!("Failed to create output array: {}", e))?;
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }
    fn solve(&mut self, a: &TensorHandle<T>, b: &TensorHandle<T>) -> Result<TensorHandle<T>> {
        let dense_a = a
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for solve (A)"))?;
        let dense_b = b
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for solve (b)"))?;
        let a_shape = dense_a.shape();
        let b_shape = dense_b.shape();
        if a_shape.len() < 2 {
            return Err(anyhow!("Matrix A must be at least 2D"));
        }
        if b_shape.is_empty() {
            return Err(anyhow!("Vector/matrix b must be at least 1D"));
        }
        let n = a_shape[a_shape.len() - 1];
        let m = a_shape[a_shape.len() - 2];
        if n != m {
            return Err(anyhow!("Matrix A must be square, got {}x{}", m, n));
        }
        let b_rows = b_shape[b_shape.len()
            - (if b_shape.len() == a_shape.len() - 1 {
                1
            } else {
                2
            })];
        if b_rows != n {
            return Err(anyhow!(
                "Dimension mismatch: A is {}x{}, b has {} rows",
                m,
                n,
                b_rows
            ));
        }
        if a_shape.len() == 2 && b_shape.len() == 1 {
            use scirs2_core::ndarray_ext::{Array1, Array2};
            let a_view = dense_a.view();
            let b_view = dense_b.view();
            let a_matrix: Array2<T> = Array2::from_shape_fn((n, n), |(i, j)| a_view[[i, j]]);
            let b_vector: Array1<T> = Array1::from_shape_fn(n, |i| b_view[[i]]);
            let x = self.solve_2d_1d(&a_matrix, &b_vector)?;
            let x_dyn = x.into_dyn();
            return Ok(TensorHandle::from_dense_auto(DenseND::from_array(x_dyn)));
        }
        Err(anyhow!(
            "Solve only supports 2D matrix A with 1D vector b in this implementation"
        ))
    }

    fn advanced_gather(
        &mut self,
        x: &TensorHandle<T>,
        axis: Axis,
        indices: &TensorHandle<T>,
        allow_negative: bool,
    ) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for advanced_gather"))?;
        let indices_dense = indices
            .as_dense()
            .ok_or_else(|| anyhow!("Indices must be dense tensor"))?;

        let result =
            super::advanced_indexing::advanced_gather(dense, axis, indices_dense, allow_negative)?;
        Ok(TensorHandle::from_dense_auto(result))
    }

    fn advanced_scatter(
        &mut self,
        shape: &[usize],
        axis: Axis,
        indices: &TensorHandle<T>,
        values: &TensorHandle<T>,
        mode: ScatterMode,
    ) -> Result<TensorHandle<T>> {
        let indices_dense = indices
            .as_dense()
            .ok_or_else(|| anyhow!("Indices must be dense tensor"))?;
        let values_dense = values
            .as_dense()
            .ok_or_else(|| anyhow!("Values must be dense tensor"))?;

        let result = super::advanced_indexing::advanced_scatter(
            shape,
            axis,
            indices_dense,
            values_dense,
            mode,
        )?;
        Ok(TensorHandle::from_dense_auto(result))
    }

    fn fancy_index_mask(
        &mut self,
        x: &TensorHandle<T>,
        mask: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>> {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for fancy_index_mask"))?;
        let mask_dense = mask
            .as_dense()
            .ok_or_else(|| anyhow!("Mask must be dense tensor"))?;

        let result = super::advanced_indexing::fancy_index_mask(dense, mask_dense)?;
        Ok(TensorHandle::from_dense_auto(result))
    }

    fn tile(&mut self, x: &TensorHandle<T>, reps: &[usize]) -> Result<TensorHandle<T>> {
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
        let mut output_data = self.acquire_pooled_generic::<T>(&output_shape);
        output_data.clear(); // Ensure buffer starts empty
        output_data.reserve(output_size);

        // Generate all output indices and map to input indices
        for i in 0..output_size {
            let out_idx = self.flat_to_multidim(i, &output_shape);
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
        self.release_pooled_generic::<T>(&output_shape, output_data);

        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }

    fn pad(
        &mut self,
        x: &TensorHandle<T>,
        pad_width: &[(usize, usize)],
        constant_value: T,
    ) -> Result<TensorHandle<T>> {
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
        let mut output_data = self.acquire_pooled_generic::<T>(&output_shape);
        output_data.clear(); // Ensure buffer starts empty
        output_data.resize(output_size, constant_value); // Initialize with constant_value

        // Copy input data to the appropriate region in output
        let input_size: usize = input_shape.iter().product();
        for i in 0..input_size {
            let in_idx = self.flat_to_multidim(i, input_shape);
            // Calculate output index by adding padding offsets
            let out_idx: Vec<usize> = in_idx
                .iter()
                .zip(pad_width.iter())
                .map(|(&idx, &(before, _))| idx + before)
                .collect();
            let out_flat = self.multidim_to_flat(&out_idx, &output_shape);
            output_data[out_flat] = input_view[in_idx.as_slice()];
        }

        use scirs2_core::ndarray_ext::{Array, IxDyn};
        let result_array = Array::from_shape_vec(IxDyn(&output_shape), output_data.clone())
            .map_err(|e| anyhow!("Failed to create padded array: {}", e))?;
        self.release_pooled_generic::<T>(&output_shape, output_data);

        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }

    fn flip(&mut self, x: &TensorHandle<T>, axes: &[Axis]) -> Result<TensorHandle<T>> {
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
        let mut output_data = self.acquire_pooled_generic::<T>(shape);
        output_data.clear(); // Ensure buffer starts empty
        output_data.reserve(total_elements);

        // For each output position, compute the corresponding flipped input position
        for i in 0..total_elements {
            let out_idx = self.flat_to_multidim(i, shape);
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
        self.release_pooled_generic::<T>(shape, output_data);

        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }

    fn squeeze(&mut self, x: &TensorHandle<T>, axes: Option<&[Axis]>) -> Result<TensorHandle<T>> {
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
        self.reshape(x, &new_shape)
    }

    fn unsqueeze(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
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

        self.reshape(x, &new_shape)
    }

    fn stack(&mut self, tensors: &[TensorHandle<T>], axis: Axis) -> Result<TensorHandle<T>> {
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
            unsqueezed.push(self.unsqueeze(tensor, axis)?);
        }

        // Concatenate along the new axis
        self.concatenate(&unsqueezed, axis)
    }

    fn repeat(
        &mut self,
        x: &TensorHandle<T>,
        repeats: usize,
        axis: Axis,
    ) -> Result<TensorHandle<T>> {
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
            let out_idx = self.flat_to_multidim(i, &output_shape);
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

    fn roll(&mut self, x: &TensorHandle<T>, shift: isize, axis: Axis) -> Result<TensorHandle<T>> {
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
            let out_idx = self.flat_to_multidim(i, shape);
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

    fn argmax(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
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
                self.flat_to_multidim(i, &output_shape)
            };

            let mut max_val = T::from_f64(f64::NEG_INFINITY).unwrap();
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

            output_data.push(T::from_usize(max_idx).unwrap());
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

    fn argmin(&mut self, x: &TensorHandle<T>, axis: Axis) -> Result<TensorHandle<T>> {
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
                self.flat_to_multidim(i, &output_shape)
            };

            let mut min_val = T::from_f64(f64::INFINITY).unwrap();
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

            output_data.push(T::from_usize(min_idx).unwrap());
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
}
