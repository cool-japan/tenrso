//! Optimized operations integration layer
//!
//! This module provides a unified interface for using the various optimization
//! modules (SIMD, tiled reductions, vectorized broadcasting) based on executor
//! configuration and tensor characteristics.
//!
//! # Strategy
//!
//! 1. Check executor configuration flags
//! 2. Check tensor characteristics (size, shape)
//! 3. Select the best implementation:
//!    - Optimized version if enabled and beneficial
//!    - Standard version otherwise
//!
//! # Performance
//!
//! The integration layer adds minimal overhead (a few boolean checks)
//! while providing significant speedups for applicable operations.

use super::simd_ops::{self, SimdBinaryOp, SimdUnaryOp};
use super::tiled_reductions;
use super::types::{BinaryOp, CpuExecutor};
use anyhow::Result;
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND};

/// Optimized unary element-wise operation
///
/// Automatically selects between SIMD and standard implementation
/// based on executor configuration and tensor size.
#[allow(dead_code)]
pub(crate) fn optimized_unary<T>(
    executor: &CpuExecutor,
    input: &DenseND<T>,
    op: UnaryOpType,
) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + FromPrimitive + Send + Sync,
{
    // Check if SIMD is enabled and tensor is large enough
    if executor.enable_simd && simd_ops::should_use_simd(input.shape()) {
        let simd_op = match op {
            UnaryOpType::Neg => SimdUnaryOp::Neg,
            UnaryOpType::Abs => SimdUnaryOp::Abs,
            UnaryOpType::Exp => SimdUnaryOp::Exp,
            UnaryOpType::Log => SimdUnaryOp::Log,
            UnaryOpType::Sin => SimdUnaryOp::Sin,
            UnaryOpType::Cos => SimdUnaryOp::Cos,
            UnaryOpType::Sqrt => SimdUnaryOp::Sqrt,
            UnaryOpType::Sqr => SimdUnaryOp::Sqr,
            UnaryOpType::Recip => SimdUnaryOp::Recip,
            UnaryOpType::Tanh => SimdUnaryOp::Tanh,
            UnaryOpType::Sigmoid => SimdUnaryOp::Sigmoid,
            UnaryOpType::ReLU => SimdUnaryOp::ReLU,
            UnaryOpType::Gelu => SimdUnaryOp::Gelu,
            UnaryOpType::Elu => SimdUnaryOp::Elu,
            UnaryOpType::Selu => SimdUnaryOp::Selu,
            UnaryOpType::Softplus => SimdUnaryOp::Softplus,
            UnaryOpType::Sign => SimdUnaryOp::Sign,
        };
        return simd_ops::simd_unary(input, simd_op);
    }

    // Fall back to standard implementation
    let result = match op {
        UnaryOpType::Neg => input.view().mapv(|v| -v),
        UnaryOpType::Abs => input.view().mapv(|v| v.abs()),
        UnaryOpType::Exp => input.view().mapv(|v| v.exp()),
        UnaryOpType::Log => input.view().mapv(|v| v.ln()),
        UnaryOpType::Sin => input.view().mapv(|v| v.sin()),
        UnaryOpType::Cos => input.view().mapv(|v| v.cos()),
        UnaryOpType::Sqrt => input.view().mapv(|v| v.sqrt()),
        UnaryOpType::Sqr => input.view().mapv(|v| v * v),
        UnaryOpType::Recip => input.view().mapv(|v| v.recip()),
        UnaryOpType::Tanh => input.view().mapv(|v| v.tanh()),
        UnaryOpType::Sigmoid => input.view().mapv(|v| {
            let one = T::one();
            one / (one + (-v).exp())
        }),
        UnaryOpType::ReLU => input.view().mapv(|v| {
            let zero = T::zero();
            if v > zero {
                v
            } else {
                zero
            }
        }),
        UnaryOpType::Gelu => input.view().mapv(|v| {
            let half = T::from_f64(0.5).unwrap_or_else(T::one);
            let one = T::one();
            let coeff = T::from_f64(0.7978845608028654).unwrap_or_else(T::one);
            let cubic_coeff = T::from_f64(0.044715).unwrap_or_else(T::zero);
            let x_cubed = v * v * v;
            let inner = coeff * (v + cubic_coeff * x_cubed);
            half * v * (one + inner.tanh())
        }),
        UnaryOpType::Elu => input.view().mapv(|v| {
            let zero = T::zero();
            let one = T::one();
            if v > zero {
                v
            } else {
                v.exp() - one
            }
        }),
        UnaryOpType::Selu => input.view().mapv(|v| {
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
        UnaryOpType::Softplus => input.view().mapv(|v| {
            let zero = T::zero();
            let one = T::one();
            let abs_v = v.abs();
            let max_part = if v > zero { v } else { zero };
            max_part + (one + (-abs_v).exp()).ln()
        }),
        UnaryOpType::Sign => input.view().mapv(|v| {
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

    Ok(DenseND::from_array(result))
}

/// Optimized binary element-wise operation
///
/// Automatically selects between SIMD, vectorized broadcasting,
/// and standard implementation based on executor configuration.
#[allow(dead_code)]
pub(crate) fn optimized_binary<T>(
    executor: &CpuExecutor,
    x: &DenseND<T>,
    y: &DenseND<T>,
    op: BinaryOp,
) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + Send + Sync + std::ops::AddAssign,
{
    // Check if shapes match (no broadcasting needed)
    if x.shape() == y.shape() {
        // Try SIMD for same-shape operations
        if executor.enable_simd && simd_ops::should_use_simd(x.shape()) {
            let simd_op = match op {
                BinaryOp::Add => SimdBinaryOp::Add,
                BinaryOp::Sub => SimdBinaryOp::Sub,
                BinaryOp::Mul => SimdBinaryOp::Mul,
                BinaryOp::Div => SimdBinaryOp::Div,
                BinaryOp::Pow => SimdBinaryOp::Pow,
                BinaryOp::Maximum => SimdBinaryOp::Maximum,
                BinaryOp::Minimum => SimdBinaryOp::Minimum,
            };
            return simd_ops::simd_binary(x, y, simd_op);
        }
    }

    // For different shapes, standard ndarray operations handle broadcasting
    use scirs2_core::ndarray_ext::Zip;
    let result = match op {
        BinaryOp::Add => &x.view() + &y.view(),
        BinaryOp::Sub => &x.view() - &y.view(),
        BinaryOp::Mul => &x.view() * &y.view(),
        BinaryOp::Div => &x.view() / &y.view(),
        BinaryOp::Pow => Zip::from(&x.view())
            .and(&y.view())
            .map_collect(|&x_val, &y_val| x_val.powf(y_val)),
        BinaryOp::Maximum => Zip::from(&x.view())
            .and(&y.view())
            .map_collect(|&x_val, &y_val| if x_val > y_val { x_val } else { y_val }),
        BinaryOp::Minimum => Zip::from(&x.view())
            .and(&y.view())
            .map_collect(|&x_val, &y_val| if x_val < y_val { x_val } else { y_val }),
    };

    Ok(DenseND::from_array(result))
}

/// Optimized sum reduction
///
/// Uses tiled reduction for large tensors when enabled.
#[allow(dead_code)]
pub(crate) fn optimized_sum<T>(
    executor: &CpuExecutor,
    input: &DenseND<T>,
    axes: &[Axis],
) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync + std::ops::AddAssign + std::iter::Sum,
{
    // For all-axes reduction (empty axes means reduce to scalar)
    if axes.is_empty() {
        if executor.enable_tiled_reductions && tiled_reductions::should_use_tiling(input.shape()) {
            let sum_val = tiled_reductions::tiled_sum_all(input)?;
            let result = scirs2_core::ndarray_ext::Array::from_elem(
                scirs2_core::ndarray_ext::IxDyn(&[]),
                sum_val,
            );
            return Ok(DenseND::from_array(result));
        } else {
            // Small tensor - use simple sum
            let sum_val: T = input.view().iter().cloned().sum();
            let result = scirs2_core::ndarray_ext::Array::from_elem(
                scirs2_core::ndarray_ext::IxDyn(&[]),
                sum_val,
            );
            return Ok(DenseND::from_array(result));
        }
    }

    // For single-axis reduction, try tiled axis reduction
    if axes.len() == 1 && executor.enable_tiled_reductions {
        return tiled_reductions::tiled_sum_axis(input, axes[0]);
    }

    // Fall back to standard implementation for multi-axis reduction
    let mut result = input.view().to_owned();
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

    for &axis_idx in &sorted_axes {
        let axis = scirs2_core::ndarray_ext::Axis(axis_idx);
        result = result.sum_axis(axis);
    }

    Ok(DenseND::from_array(result))
}

/// Optimized mean reduction
///
/// Uses tiled reduction for large tensors when enabled.
#[allow(dead_code)]
pub(crate) fn optimized_mean<T>(
    executor: &CpuExecutor,
    input: &DenseND<T>,
    axes: &[Axis],
) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync + std::ops::AddAssign + Float + FromPrimitive + std::iter::Sum,
{
    // For all-axes reduction (empty axes means reduce to scalar)
    if axes.is_empty() {
        if executor.enable_tiled_reductions && tiled_reductions::should_use_tiling(input.shape()) {
            let mean_val = tiled_reductions::tiled_mean_all(input)?;
            let result = scirs2_core::ndarray_ext::Array::from_elem(
                scirs2_core::ndarray_ext::IxDyn(&[]),
                mean_val,
            );
            return Ok(DenseND::from_array(result));
        } else {
            // Small tensor - use simple mean
            let total_elements = input.view().len();
            let sum: T = input.view().iter().cloned().sum();
            let mean = sum / T::from_usize(total_elements).unwrap();
            let result = scirs2_core::ndarray_ext::Array::from_elem(
                scirs2_core::ndarray_ext::IxDyn(&[]),
                mean,
            );
            return Ok(DenseND::from_array(result));
        }
    }

    // Fall back to standard implementation
    let mut result = input.view().to_owned();
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

    for &axis_idx in &sorted_axes {
        let axis = scirs2_core::ndarray_ext::Axis(axis_idx);
        result = result
            .mean_axis(axis)
            .ok_or_else(|| anyhow::anyhow!("Mean computation failed"))?;
    }

    Ok(DenseND::from_array(result))
}

/// Unary operation types for optimized dispatch
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) enum UnaryOpType {
    Neg,
    Abs,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Sqr,
    Recip,
    Tanh,
    Sigmoid,
    ReLU,
    Gelu,
    Elu,
    Selu,
    Softplus,
    Sign,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_unary_small_tensor() {
        let executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = optimized_unary(&executor, &input, UnaryOpType::Neg).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], -1.0);
        assert_eq!(result_view[[1]], -2.0);
        assert_eq!(result_view[[2]], -3.0);
        assert_eq!(result_view[[3]], -4.0);
    }

    #[test]
    fn test_optimized_binary_same_shape() {
        let executor = CpuExecutor::new();
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();

        let result = optimized_binary(&executor, &a, &b, BinaryOp::Add).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 6.0);
        assert_eq!(result_view[[1]], 8.0);
        assert_eq!(result_view[[2]], 10.0);
        assert_eq!(result_view[[3]], 12.0);
    }

    #[test]
    fn test_optimized_sum_all() {
        let executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        let result = optimized_sum(&executor, &input, &[]).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[]], 15.0);
    }

    #[test]
    fn test_optimized_mean_all() {
        let executor = CpuExecutor::new();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        let result = optimized_mean(&executor, &input, &[]).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[]], 3.0);
    }

    #[test]
    fn test_optimization_disabled() {
        let executor = CpuExecutor::unoptimized();
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Should still work, just without optimizations
        let result = optimized_unary(&executor, &input, UnaryOpType::Exp).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] - std::f64::consts::E).abs() < 1e-10);
    }
}
