//! Element-wise operations for tensors
//!
//! This module provides element-wise unary and binary operations including:
//! - Tensor-scalar binary operations (add, sub, mul, div, pow)
//! - Full-reduction support for reduction operations
//! - Parallel dispatch for large tensors
//!
//! These operations integrate with the CpuExecutor and respect its
//! configuration for SIMD, parallel execution, and memory pooling.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{DenseND, TensorHandle};

use super::parallel::should_parallelize;
use super::types::{BinaryOp, CpuExecutor, ElemOp, ReduceOp};

/// Scalar operation types for tensor-scalar binary ops
#[derive(Clone, Debug)]
pub enum ScalarOp {
    /// Add scalar to each element: x + s
    Add,
    /// Subtract scalar from each element: x - s
    Sub,
    /// Multiply each element by scalar: x * s
    Mul,
    /// Divide each element by scalar: x / s
    Div,
    /// Raise each element to scalar power: x^s
    Pow,
}

impl CpuExecutor {
    /// Apply a scalar binary operation to a tensor
    ///
    /// Computes `op(tensor_element, scalar)` for each element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `op` - The scalar operation to apply
    /// * `x` - Input tensor handle
    /// * `scalar` - Scalar value to combine with each element
    ///
    /// # Returns
    ///
    /// A new tensor handle with the same shape as `x`, containing the results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not dense
    /// - Division by zero is attempted (for ScalarOp::Div)
    pub fn scalar_op<T>(
        &mut self,
        op: ScalarOp,
        x: &TensorHandle<T>,
        scalar: T,
    ) -> Result<TensorHandle<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for scalar_op"))?;

        if let ScalarOp::Div = op {
            if scalar == T::zero() {
                return Err(anyhow!("Division by zero in scalar division"));
            }
        }

        let use_parallel = self.enable_parallel && should_parallelize(dense.shape());

        let result_data = if use_parallel {
            self.parallel_scalar_op_inner(&op, dense, scalar)
        } else {
            self.serial_scalar_op_inner(&op, dense, scalar)
        };

        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_data,
        )))
    }

    /// Serial implementation of scalar operations
    fn serial_scalar_op_inner<T>(
        &self,
        op: &ScalarOp,
        dense: &DenseND<T>,
        scalar: T,
    ) -> Array<T, IxDyn>
    where
        T: Clone + Num + Float + FromPrimitive,
    {
        match op {
            ScalarOp::Add => dense.view().mapv(|v| v + scalar),
            ScalarOp::Sub => dense.view().mapv(|v| v - scalar),
            ScalarOp::Mul => dense.view().mapv(|v| v * scalar),
            ScalarOp::Div => dense.view().mapv(|v| v / scalar),
            ScalarOp::Pow => dense.view().mapv(|v| v.powf(scalar)),
        }
    }

    /// Parallel implementation of scalar operations using rayon
    fn parallel_scalar_op_inner<T>(
        &self,
        op: &ScalarOp,
        dense: &DenseND<T>,
        scalar: T,
    ) -> Array<T, IxDyn>
    where
        T: Clone + Num + Float + FromPrimitive + Send + Sync,
    {
        use rayon::prelude::*;

        let input_data: Vec<T> = dense.view().iter().cloned().collect();
        let result_data: Vec<T> = match op {
            ScalarOp::Add => input_data.par_iter().map(|&v| v + scalar).collect(),
            ScalarOp::Sub => input_data.par_iter().map(|&v| v - scalar).collect(),
            ScalarOp::Mul => input_data.par_iter().map(|&v| v * scalar).collect(),
            ScalarOp::Div => input_data.par_iter().map(|&v| v / scalar).collect(),
            ScalarOp::Pow => input_data.par_iter().map(|&v| v.powf(scalar)).collect(),
        };

        Array::from_shape_vec(IxDyn(dense.shape()), result_data)
            .unwrap_or_else(|_| dense.view().mapv(|v| v + scalar))
    }

    /// Apply an element-wise unary operation with parallel dispatch
    ///
    /// This method automatically selects between serial and parallel execution
    /// based on tensor size and executor configuration.
    ///
    /// # Arguments
    ///
    /// * `op` - The unary element-wise operation to apply
    /// * `x` - Input tensor handle
    ///
    /// # Returns
    ///
    /// A new tensor handle with the same shape, containing the results.
    pub fn parallel_elem_op<T>(
        &mut self,
        op: ElemOp,
        x: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for parallel_elem_op"))?;

        let use_parallel = self.enable_parallel && should_parallelize(dense.shape());

        let result_data = if use_parallel {
            self.parallel_elem_op_inner(&op, dense)
        } else {
            self.serial_elem_op_inner(&op, dense)
        };

        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_data,
        )))
    }

    /// Serial unary op implementation
    fn serial_elem_op_inner<T>(&self, op: &ElemOp, dense: &DenseND<T>) -> Array<T, IxDyn>
    where
        T: Clone + Num + Float + FromPrimitive,
    {
        match op {
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
            ElemOp::Sigmoid => {
                let one = T::one();
                dense.view().mapv(|v| one / (one + (-v).exp()))
            }
            ElemOp::ReLU => {
                let zero = T::zero();
                dense.view().mapv(|v| if v > zero { v } else { zero })
            }
            ElemOp::Gelu => {
                let half = T::from_f64(0.5).unwrap_or_else(T::one);
                let one = T::one();
                let coeff = T::from_f64(0.797_884_560_802_865_4).unwrap_or_else(T::one);
                let cubic_coeff = T::from_f64(0.044715).unwrap_or_else(T::zero);
                dense.view().mapv(|v| {
                    let x_cubed = v * v * v;
                    let inner = coeff * (v + cubic_coeff * x_cubed);
                    half * v * (one + inner.tanh())
                })
            }
            ElemOp::Elu => {
                let zero = T::zero();
                let one = T::one();
                dense
                    .view()
                    .mapv(|v| if v > zero { v } else { v.exp() - one })
            }
            ElemOp::Selu => {
                let zero = T::zero();
                let one = T::one();
                let scale = T::from_f64(1.050_700_987_355_480_5).unwrap_or_else(T::one);
                let alpha = T::from_f64(1.673_263_242_354_377_2).unwrap_or_else(T::one);
                dense.view().mapv(|v| {
                    if v > zero {
                        scale * v
                    } else {
                        scale * alpha * (v.exp() - one)
                    }
                })
            }
            ElemOp::Softplus => {
                let zero = T::zero();
                let one = T::one();
                dense.view().mapv(|v| {
                    let abs_v = v.abs();
                    let max_part = if v > zero { v } else { zero };
                    max_part + (one + (-abs_v).exp()).ln()
                })
            }
            ElemOp::Sign => {
                let zero = T::zero();
                let one = T::one();
                let neg_one = -one;
                dense.view().mapv(|v| {
                    if v > zero {
                        one
                    } else if v < zero {
                        neg_one
                    } else {
                        zero
                    }
                })
            }
        }
    }

    /// Parallel unary op implementation using rayon
    fn parallel_elem_op_inner<T>(&self, op: &ElemOp, dense: &DenseND<T>) -> Array<T, IxDyn>
    where
        T: Clone + Num + Float + FromPrimitive + Send + Sync,
    {
        use rayon::prelude::*;

        let input_data: Vec<T> = dense.view().iter().cloned().collect();
        let result_data: Vec<T> = match op {
            ElemOp::Neg => input_data.par_iter().map(|&v| -v).collect(),
            ElemOp::Abs => input_data.par_iter().map(|&v| v.abs()).collect(),
            ElemOp::Exp => input_data.par_iter().map(|&v| v.exp()).collect(),
            ElemOp::Log => input_data.par_iter().map(|&v| v.ln()).collect(),
            ElemOp::Sin => input_data.par_iter().map(|&v| v.sin()).collect(),
            ElemOp::Cos => input_data.par_iter().map(|&v| v.cos()).collect(),
            ElemOp::Sqrt => input_data.par_iter().map(|&v| v.sqrt()).collect(),
            ElemOp::Sqr => input_data.par_iter().map(|&v| v * v).collect(),
            ElemOp::Recip => input_data.par_iter().map(|&v| v.recip()).collect(),
            ElemOp::Tanh => input_data.par_iter().map(|&v| v.tanh()).collect(),
            ElemOp::Sigmoid => {
                let one = T::one();
                input_data
                    .par_iter()
                    .map(|&v| one / (one + (-v).exp()))
                    .collect()
            }
            ElemOp::ReLU => {
                let zero = T::zero();
                input_data
                    .par_iter()
                    .map(|&v| if v > zero { v } else { zero })
                    .collect()
            }
            ElemOp::Gelu => {
                let half = T::from_f64(0.5).unwrap_or_else(T::one);
                let one = T::one();
                let coeff = T::from_f64(0.797_884_560_802_865_4).unwrap_or_else(T::one);
                let cubic_coeff = T::from_f64(0.044715).unwrap_or_else(T::zero);
                input_data
                    .par_iter()
                    .map(|&v| {
                        let x_cubed = v * v * v;
                        let inner = coeff * (v + cubic_coeff * x_cubed);
                        half * v * (one + inner.tanh())
                    })
                    .collect()
            }
            ElemOp::Elu => {
                let zero = T::zero();
                let one = T::one();
                input_data
                    .par_iter()
                    .map(|&v| if v > zero { v } else { v.exp() - one })
                    .collect()
            }
            ElemOp::Selu => {
                let zero = T::zero();
                let one = T::one();
                let scale = T::from_f64(1.050_700_987_355_480_5).unwrap_or_else(T::one);
                let alpha = T::from_f64(1.673_263_242_354_377_2).unwrap_or_else(T::one);
                input_data
                    .par_iter()
                    .map(|&v| {
                        if v > zero {
                            scale * v
                        } else {
                            scale * alpha * (v.exp() - one)
                        }
                    })
                    .collect()
            }
            ElemOp::Softplus => {
                let zero = T::zero();
                let one = T::one();
                input_data
                    .par_iter()
                    .map(|&v| {
                        let abs_v = v.abs();
                        let max_part = if v > zero { v } else { zero };
                        max_part + (one + (-abs_v).exp()).ln()
                    })
                    .collect()
            }
            ElemOp::Sign => {
                let zero = T::zero();
                let one = T::one();
                let neg_one = -one;
                input_data
                    .par_iter()
                    .map(|&v| {
                        if v > zero {
                            one
                        } else if v < zero {
                            neg_one
                        } else {
                            zero
                        }
                    })
                    .collect()
            }
        };

        Array::from_shape_vec(IxDyn(dense.shape()), result_data)
            .unwrap_or_else(|_| dense.view().to_owned())
    }

    /// Apply a binary operation with parallel dispatch
    ///
    /// Automatically selects between serial and parallel execution
    /// based on tensor size and executor configuration.
    ///
    /// # Arguments
    ///
    /// * `op` - The binary element-wise operation
    /// * `x` - First input tensor
    /// * `y` - Second input tensor
    ///
    /// # Returns
    ///
    /// Result tensor with same shape (or broadcasted shape).
    pub fn parallel_binary_op<T>(
        &mut self,
        op: BinaryOp,
        x: &TensorHandle<T>,
        y: &TensorHandle<T>,
    ) -> Result<TensorHandle<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let dense_x = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for parallel_binary_op"))?;
        let dense_y = y
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for parallel_binary_op"))?;

        // Only parallelize same-shape operations for now
        if dense_x.shape() != dense_y.shape() {
            return self.binary_op_with_broadcast(op, dense_x, dense_y);
        }

        let use_parallel = self.enable_parallel && should_parallelize(dense_x.shape());

        if use_parallel {
            self.parallel_binary_inner(&op, dense_x, dense_y)
        } else {
            // Delegate to standard implementation
            use scirs2_core::ndarray_ext::Zip;
            let result_data = match op {
                BinaryOp::Add => &dense_x.view() + &dense_y.view(),
                BinaryOp::Sub => &dense_x.view() - &dense_y.view(),
                BinaryOp::Mul => &dense_x.view() * &dense_y.view(),
                BinaryOp::Div => &dense_x.view() / &dense_y.view(),
                BinaryOp::Pow => Zip::from(&dense_x.view())
                    .and(&dense_y.view())
                    .map_collect(|&x_val, &y_val| x_val.powf(y_val)),
                BinaryOp::Maximum => Zip::from(&dense_x.view())
                    .and(&dense_y.view())
                    .map_collect(|&x_val, &y_val| if x_val > y_val { x_val } else { y_val }),
                BinaryOp::Minimum => Zip::from(&dense_x.view())
                    .and(&dense_y.view())
                    .map_collect(|&x_val, &y_val| if x_val < y_val { x_val } else { y_val }),
            };
            Ok(TensorHandle::from_dense_auto(DenseND::from_array(
                result_data,
            )))
        }
    }

    /// Parallel binary op implementation using rayon
    fn parallel_binary_inner<T>(
        &self,
        op: &BinaryOp,
        x: &DenseND<T>,
        y: &DenseND<T>,
    ) -> Result<TensorHandle<T>>
    where
        T: Clone + Num + Float + FromPrimitive + Send + Sync,
    {
        use rayon::prelude::*;

        let x_data: Vec<T> = x.view().iter().cloned().collect();
        let y_data: Vec<T> = y.view().iter().cloned().collect();

        let result_data: Vec<T> = x_data
            .par_iter()
            .zip(y_data.par_iter())
            .map(|(&xv, &yv)| match op {
                BinaryOp::Add => xv + yv,
                BinaryOp::Sub => xv - yv,
                BinaryOp::Mul => xv * yv,
                BinaryOp::Div => xv / yv,
                BinaryOp::Pow => xv.powf(yv),
                BinaryOp::Maximum => {
                    if xv > yv {
                        xv
                    } else {
                        yv
                    }
                }
                BinaryOp::Minimum => {
                    if xv < yv {
                        xv
                    } else {
                        yv
                    }
                }
            })
            .collect();

        let result_array = Array::from_shape_vec(IxDyn(x.shape()), result_data)
            .map_err(|e| anyhow!("Failed to create result array: {}", e))?;
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }

    /// Full reduction (reduce all elements to a scalar)
    ///
    /// This method reduces all elements of a tensor to a single scalar value,
    /// regardless of the tensor's shape.
    ///
    /// # Arguments
    ///
    /// * `op` - The reduction operation to apply
    /// * `x` - Input tensor handle
    ///
    /// # Returns
    ///
    /// A scalar tensor (0-dimensional) containing the reduction result.
    pub fn full_reduce<T>(&mut self, op: ReduceOp, x: &TensorHandle<T>) -> Result<TensorHandle<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for full_reduce"))?;

        let view = dense.view();
        let total_elements = view.len();

        if total_elements == 0 {
            return Err(anyhow!("Cannot reduce empty tensor"));
        }

        let use_parallel = self.enable_parallel && should_parallelize(dense.shape());

        let result_val = if use_parallel {
            self.parallel_full_reduce_inner(&op, dense)?
        } else {
            self.serial_full_reduce_inner(&op, dense)?
        };

        let result_array = Array::from_elem(IxDyn(&[]), result_val);
        Ok(TensorHandle::from_dense_auto(DenseND::from_array(
            result_array,
        )))
    }

    /// Serial full reduction implementation
    fn serial_full_reduce_inner<T>(&self, op: &ReduceOp, dense: &DenseND<T>) -> Result<T>
    where
        T: Clone + Num + std::ops::AddAssign + std::default::Default + Float + FromPrimitive,
    {
        let view = dense.view();
        let total_elements = view.len();

        match op {
            ReduceOp::Sum => {
                let mut sum = T::zero();
                for &v in view.iter() {
                    sum += v;
                }
                Ok(sum)
            }
            ReduceOp::Mean => {
                let mut sum = T::zero();
                for &v in view.iter() {
                    sum += v;
                }
                let count = T::from_usize(total_elements)
                    .ok_or_else(|| anyhow!("Failed to convert element count to type T"))?;
                Ok(sum / count)
            }
            ReduceOp::Max => {
                let result = view
                    .iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| anyhow!("Empty tensor for max reduction"))?;
                Ok(result)
            }
            ReduceOp::Min => {
                let result = view
                    .iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| anyhow!("Empty tensor for min reduction"))?;
                Ok(result)
            }
            ReduceOp::Prod => {
                let result = view.iter().cloned().fold(T::one(), |acc, x| acc * x);
                Ok(result)
            }
            ReduceOp::All => {
                let all_nonzero = view.iter().all(|&x| x != T::zero());
                Ok(if all_nonzero { T::one() } else { T::zero() })
            }
            ReduceOp::Any => {
                let any_nonzero = view.iter().any(|&x| x != T::zero());
                Ok(if any_nonzero { T::one() } else { T::zero() })
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin => Err(anyhow!(
                "ArgMax/ArgMin should use dedicated argmax/argmin methods"
            )),
        }
    }

    /// Parallel full reduction implementation using rayon
    fn parallel_full_reduce_inner<T>(&self, op: &ReduceOp, dense: &DenseND<T>) -> Result<T>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + Send
            + Sync,
    {
        use rayon::prelude::*;

        let data: Vec<T> = dense.view().iter().cloned().collect();
        let total_elements = data.len();

        match op {
            ReduceOp::Sum => {
                let sum = data.par_iter().cloned().reduce(|| T::zero(), |a, b| a + b);
                Ok(sum)
            }
            ReduceOp::Mean => {
                let sum = data.par_iter().cloned().reduce(|| T::zero(), |a, b| a + b);
                let count = T::from_usize(total_elements)
                    .ok_or_else(|| anyhow!("Failed to convert element count to type T"))?;
                Ok(sum / count)
            }
            ReduceOp::Max => {
                let result = data
                    .par_iter()
                    .cloned()
                    .reduce(|| T::neg_infinity(), |a, b| if a > b { a } else { b });
                Ok(result)
            }
            ReduceOp::Min => {
                let result = data
                    .par_iter()
                    .cloned()
                    .reduce(|| T::infinity(), |a, b| if a < b { a } else { b });
                Ok(result)
            }
            ReduceOp::Prod => {
                let result = data.par_iter().cloned().reduce(|| T::one(), |a, b| a * b);
                Ok(result)
            }
            ReduceOp::All => {
                let all_nonzero = data.par_iter().all(|&x| x != T::zero());
                Ok(if all_nonzero { T::one() } else { T::zero() })
            }
            ReduceOp::Any => {
                let any_nonzero = data.par_iter().any(|&x| x != T::zero());
                Ok(if any_nonzero { T::one() } else { T::zero() })
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin => Err(anyhow!(
                "ArgMax/ArgMin should use dedicated argmax/argmin methods"
            )),
        }
    }

    /// Parallel reduction along specified axes
    ///
    /// Like `reduce()` but with automatic parallel dispatch for large tensors.
    /// Also supports empty axes (full reduction to scalar).
    ///
    /// # Arguments
    ///
    /// * `op` - The reduction operation
    /// * `x` - Input tensor handle
    /// * `axes` - Axes along which to reduce. Empty = full reduction.
    ///
    /// # Returns
    ///
    /// Reduced tensor (scalar if axes is empty, reduced-dimension otherwise).
    pub fn parallel_reduce<T>(
        &mut self,
        op: ReduceOp,
        x: &TensorHandle<T>,
        axes: &[usize],
    ) -> Result<TensorHandle<T>>
    where
        T: Clone
            + Num
            + std::ops::AddAssign
            + std::default::Default
            + Float
            + FromPrimitive
            + Send
            + Sync
            + 'static,
    {
        if axes.is_empty() {
            return self.full_reduce(op, x);
        }

        let dense = x
            .as_dense()
            .ok_or_else(|| anyhow!("Only dense tensors supported for parallel_reduce"))?;

        let ndim = dense.shape().len();
        for &axis_idx in axes {
            if axis_idx >= ndim {
                return Err(anyhow!(
                    "Axis index {} out of range for tensor with {} dimensions",
                    axis_idx,
                    ndim
                ));
            }
        }

        // For axis-based reduction, use the standard approach
        // (ndarray's sum_axis etc. are already reasonably parallel-friendly)
        let mut result = dense.view().to_owned();
        let mut sorted_axes = axes.to_vec();
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
                ReduceOp::Mean => result
                    .mean_axis(axis)
                    .ok_or_else(|| anyhow!("Mean reduction failed"))?,
                ReduceOp::Prod => result.map_axis(axis, |view| {
                    view.iter().cloned().fold(T::one(), |acc, x| acc * x)
                }),
                ReduceOp::All => result.map_axis(axis, |view| {
                    if view.iter().all(|&x| x != T::zero()) {
                        T::one()
                    } else {
                        T::zero()
                    }
                }),
                ReduceOp::Any => result.map_axis(axis, |view| {
                    if view.iter().any(|&x| x != T::zero()) {
                        T::one()
                    } else {
                        T::zero()
                    }
                }),
                ReduceOp::ArgMax | ReduceOp::ArgMin => {
                    return Err(anyhow!(
                        "ArgMax/ArgMin should use dedicated argmax/argmin methods"
                    ));
                }
            };
        }

        Ok(TensorHandle::from_dense_auto(DenseND::from_array(result)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenrso_core::DenseND;

    // ========================================================================
    // Scalar operation tests
    // ========================================================================

    #[test]
    fn test_scalar_add() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .scalar_op(ScalarOp::Add, &handle, 10.0)
            .expect("scalar_op add failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0, 0]] - 11.0_f64).abs() < 1e-10);
        assert!((view[[0, 1]] - 12.0_f64).abs() < 1e-10);
        assert!((view[[1, 0]] - 13.0_f64).abs() < 1e-10);
        assert!((view[[1, 1]] - 14.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_sub() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .scalar_op(ScalarOp::Sub, &handle, 5.0)
            .expect("scalar_op sub failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 5.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 15.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 25.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_mul() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .scalar_op(ScalarOp::Mul, &handle, 3.0)
            .expect("scalar_op mul failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 3.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 6.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 9.0_f64).abs() < 1e-10);
        assert!((view[[3]] - 12.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_div() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .scalar_op(ScalarOp::Div, &handle, 5.0)
            .expect("scalar_op div failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 2.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 4.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 6.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_div_by_zero() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0], &[2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor.scalar_op(ScalarOp::Div, &handle, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar_pow() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![2.0, 3.0, 4.0], &[3]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .scalar_op(ScalarOp::Pow, &handle, 2.0)
            .expect("scalar_op pow failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 4.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 9.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 16.0_f64).abs() < 1e-10);
    }

    // ========================================================================
    // Full reduction tests
    // ========================================================================

    #[test]
    fn test_full_reduce_sum() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .full_reduce(ReduceOp::Sum, &handle)
            .expect("full_reduce sum failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 21.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_full_reduce_mean() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .full_reduce(ReduceOp::Mean, &handle)
            .expect("full_reduce mean failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 5.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_full_reduce_max() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![3.0, 1.0, 7.0, 2.0, 9.0, 4.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .full_reduce(ReduceOp::Max, &handle)
            .expect("full_reduce max failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 9.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_full_reduce_min() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![3.0, 1.0, 7.0, 2.0, 9.0, 4.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .full_reduce(ReduceOp::Min, &handle)
            .expect("full_reduce min failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 1.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_full_reduce_prod() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .full_reduce(ReduceOp::Prod, &handle)
            .expect("full_reduce prod failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 24.0_f64).abs() < 1e-10);
    }

    // ========================================================================
    // Parallel reduction tests
    // ========================================================================

    #[test]
    fn test_parallel_reduce_empty_axes_is_full_reduce() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_reduce(ReduceOp::Sum, &handle, &[])
            .expect("parallel_reduce sum failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 21.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce_along_axis_0() {
        let mut executor = CpuExecutor::new();
        // [[1, 2, 3], [4, 5, 6]] -> sum along axis 0 -> [5, 7, 9]
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_reduce(ReduceOp::Sum, &handle, &[0])
            .expect("parallel_reduce sum axis 0 failed");
        let dense = result.as_dense().expect("as_dense failed");
        assert_eq!(dense.shape(), &[3]);
        let view = dense.view();
        assert!((view[[0]] - 5.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 7.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 9.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce_along_axis_1() {
        let mut executor = CpuExecutor::new();
        // [[1, 2, 3], [4, 5, 6]] -> mean along axis 1 -> [2, 5]
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_reduce(ReduceOp::Mean, &handle, &[1])
            .expect("parallel_reduce mean axis 1 failed");
        let dense = result.as_dense().expect("as_dense failed");
        assert_eq!(dense.shape(), &[2]);
        let view = dense.view();
        assert!((view[[0]] - 2.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 5.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce_max_along_axis() {
        let mut executor = CpuExecutor::new();
        // [[1, 5, 3], [4, 2, 6]] -> max along axis 0 -> [4, 5, 6]
        let data = DenseND::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_reduce(ReduceOp::Max, &handle, &[0])
            .expect("parallel_reduce max axis 0 failed");
        let dense = result.as_dense().expect("as_dense failed");
        assert_eq!(dense.shape(), &[3]);
        let view = dense.view();
        assert!((view[[0]] - 4.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 5.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 6.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce_invalid_axis() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor.parallel_reduce(ReduceOp::Sum, &handle, &[5]);
        assert!(result.is_err());
    }

    // ========================================================================
    // Parallel element-wise operation tests
    // ========================================================================

    #[test]
    fn test_parallel_elem_op_neg() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[4]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_elem_op(ElemOp::Neg, &handle)
            .expect("parallel_elem_op neg failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - (-1.0_f64)).abs() < 1e-10);
        assert!((view[[1]] - 2.0_f64).abs() < 1e-10);
        assert!((view[[2]] - (-3.0_f64)).abs() < 1e-10);
        assert!((view[[3]] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_elem_op_exp() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![0.0, 1.0, 2.0], &[3]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_elem_op(ElemOp::Exp, &handle)
            .expect("parallel_elem_op exp failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((view[[1]] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_elem_op_sqrt() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_elem_op(ElemOp::Sqrt, &handle)
            .expect("parallel_elem_op sqrt failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 1.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 2.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 3.0_f64).abs() < 1e-10);
        assert!((view[[3]] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_elem_op_abs() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_elem_op(ElemOp::Abs, &handle)
            .expect("parallel_elem_op abs failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0, 0]] - 1.0_f64).abs() < 1e-10);
        assert!((view[[0, 1]] - 2.0_f64).abs() < 1e-10);
        assert!((view[[1, 0]] - 3.0_f64).abs() < 1e-10);
        assert!((view[[1, 1]] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_elem_op_log() {
        let mut executor = CpuExecutor::new();
        let data =
            DenseND::from_vec(vec![1.0, std::f64::consts::E], &[2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .parallel_elem_op(ElemOp::Log, &handle)
            .expect("parallel_elem_op log failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 0.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 1.0_f64).abs() < 1e-10);
    }

    // ========================================================================
    // Parallel binary operation tests
    // ========================================================================

    #[test]
    fn test_parallel_binary_add() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("from_vec failed");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).expect("from_vec failed");
        let ha = TensorHandle::from_dense_auto(a);
        let hb = TensorHandle::from_dense_auto(b);

        let result = executor
            .parallel_binary_op(BinaryOp::Add, &ha, &hb)
            .expect("parallel_binary_op add failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 6.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 8.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 10.0_f64).abs() < 1e-10);
        assert!((view[[3]] - 12.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_binary_mul() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![2.0, 3.0, 4.0], &[3]).expect("from_vec failed");
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0], &[3]).expect("from_vec failed");
        let ha = TensorHandle::from_dense_auto(a);
        let hb = TensorHandle::from_dense_auto(b);

        let result = executor
            .parallel_binary_op(BinaryOp::Mul, &ha, &hb)
            .expect("parallel_binary_op mul failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 10.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 18.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 28.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_binary_sub() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).expect("from_vec failed");
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("from_vec failed");
        let ha = TensorHandle::from_dense_auto(a);
        let hb = TensorHandle::from_dense_auto(b);

        let result = executor
            .parallel_binary_op(BinaryOp::Sub, &ha, &hb)
            .expect("parallel_binary_op sub failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 9.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 18.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 27.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_binary_div() {
        let mut executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![10.0, 20.0, 30.0], &[3]).expect("from_vec failed");
        let b = DenseND::from_vec(vec![2.0, 4.0, 5.0], &[3]).expect("from_vec failed");
        let ha = TensorHandle::from_dense_auto(a);
        let hb = TensorHandle::from_dense_auto(b);

        let result = executor
            .parallel_binary_op(BinaryOp::Div, &ha, &hb)
            .expect("parallel_binary_op div failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[0]] - 5.0_f64).abs() < 1e-10);
        assert!((view[[1]] - 5.0_f64).abs() < 1e-10);
        assert!((view[[2]] - 6.0_f64).abs() < 1e-10);
    }

    // ========================================================================
    // Parallel consistency tests (serial vs parallel produce same results)
    // ========================================================================

    #[test]
    fn test_parallel_vs_serial_scalar_add() {
        let data_vals: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

        let mut executor_serial = CpuExecutor::serial();
        let mut executor_parallel = CpuExecutor::new();

        let data_s = DenseND::from_vec(data_vals.clone(), &[10, 10]).expect("from_vec failed");
        let data_p = DenseND::from_vec(data_vals, &[10, 10]).expect("from_vec failed");
        let handle_s = TensorHandle::from_dense_auto(data_s);
        let handle_p = TensorHandle::from_dense_auto(data_p);

        let result_s = executor_serial
            .scalar_op(ScalarOp::Add, &handle_s, 42.0)
            .expect("serial scalar_op failed");
        let result_p = executor_parallel
            .scalar_op(ScalarOp::Add, &handle_p, 42.0)
            .expect("parallel scalar_op failed");

        let dense_s = result_s.as_dense().expect("as_dense failed");
        let dense_p = result_p.as_dense().expect("as_dense failed");
        let view_s = dense_s.view();
        let view_p = dense_p.view();

        for i in 0..10 {
            for j in 0..10 {
                let diff: f64 = (view_s[[i, j]] - view_p[[i, j]]).abs();
                assert!(diff < 1e-10, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_parallel_vs_serial_full_reduce_sum() {
        let data_vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        let mut executor_serial = CpuExecutor::serial();
        let mut executor_parallel = CpuExecutor::new();

        let data_s = DenseND::from_vec(data_vals.clone(), &[10, 10]).expect("from_vec failed");
        let data_p = DenseND::from_vec(data_vals, &[10, 10]).expect("from_vec failed");
        let handle_s = TensorHandle::from_dense_auto(data_s);
        let handle_p = TensorHandle::from_dense_auto(data_p);

        let result_s = executor_serial
            .full_reduce(ReduceOp::Sum, &handle_s)
            .expect("serial full_reduce failed");
        let result_p = executor_parallel
            .full_reduce(ReduceOp::Sum, &handle_p)
            .expect("parallel full_reduce failed");

        let dense_s = result_s.as_dense().expect("as_dense failed");
        let dense_p = result_p.as_dense().expect("as_dense failed");
        let view_s = dense_s.view();
        let view_p = dense_p.view();

        // Sum of 1..=100 = 5050
        assert!((view_s[[]] - 5050.0_f64).abs() < 1e-10);
        assert!((view_p[[]] - 5050.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_vs_serial_elem_op() {
        let data_vals: Vec<f64> = (1..=20).map(|i| i as f64).collect();

        let mut executor_serial = CpuExecutor::serial();
        let mut executor_parallel = CpuExecutor::new();

        let data_s = DenseND::from_vec(data_vals.clone(), &[4, 5]).expect("from_vec failed");
        let data_p = DenseND::from_vec(data_vals, &[4, 5]).expect("from_vec failed");
        let handle_s = TensorHandle::from_dense_auto(data_s);
        let handle_p = TensorHandle::from_dense_auto(data_p);

        let result_s = executor_serial
            .parallel_elem_op(ElemOp::Sqrt, &handle_s)
            .expect("serial parallel_elem_op failed");
        let result_p = executor_parallel
            .parallel_elem_op(ElemOp::Sqrt, &handle_p)
            .expect("parallel parallel_elem_op failed");

        let dense_s = result_s.as_dense().expect("as_dense failed");
        let dense_p = result_p.as_dense().expect("as_dense failed");

        for (vs, vp) in dense_s.view().iter().zip(dense_p.view().iter()) {
            let diff: f64 = (vs - vp).abs();
            assert!(diff < 1e-10);
        }
    }

    // ========================================================================
    // Scalar op on 3D tensor
    // ========================================================================

    #[test]
    fn test_scalar_op_3d_tensor() {
        let mut executor = CpuExecutor::new();
        let data = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])
            .expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .scalar_op(ScalarOp::Mul, &handle, 2.0)
            .expect("scalar_op mul failed");
        let dense = result.as_dense().expect("as_dense failed");
        assert_eq!(dense.shape(), &[2, 2, 2]);
        let view = dense.view();
        assert!((view[[0, 0, 0]] - 2.0_f64).abs() < 1e-10);
        assert!((view[[1, 1, 1]] - 16.0_f64).abs() < 1e-10);
    }

    // ========================================================================
    // Full reduce on 3D tensor
    // ========================================================================

    #[test]
    fn test_full_reduce_3d() {
        let mut executor = CpuExecutor::new();
        // 2x2x2 tensor, all ones -> sum = 8
        let data = DenseND::from_vec(vec![1.0; 8], &[2, 2, 2]).expect("from_vec failed");
        let handle = TensorHandle::from_dense_auto(data);

        let result = executor
            .full_reduce(ReduceOp::Sum, &handle)
            .expect("full_reduce sum failed");
        let dense = result.as_dense().expect("as_dense failed");
        let view = dense.view();
        assert!((view[[]] - 8.0_f64).abs() < 1e-10);
    }
}
