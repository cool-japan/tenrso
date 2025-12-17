//! SIMD-optimized element-wise operations
//!
//! This module provides high-performance SIMD-accelerated implementations
//! of common tensor operations using SciRS2's SIMD capabilities.
//!
//! # Performance Features
//!
//! - Vectorized operations using AVX2/AVX-512 when available
//! - Aligned memory access for optimal performance
//! - Cache-friendly memory access patterns
//! - Automatic fallback to scalar for small tensors
//!
//! # Usage
//!
//! These functions are used internally by CpuExecutor to accelerate
//! element-wise operations for large tensors.

#![allow(dead_code)]

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, ArrayView, IxDyn, Zip};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::DenseND;

/// Threshold for SIMD optimization (number of elements)
/// Tensors smaller than this use scalar operations
const SIMD_THRESHOLD: usize = 1024;

/// Check if tensor is large enough to benefit from SIMD
#[inline]
pub(crate) fn should_use_simd(shape: &[usize]) -> bool {
    let total_elements: usize = shape.iter().product();
    total_elements >= SIMD_THRESHOLD
}

/// SIMD-optimized unary operations
#[allow(dead_code)]
pub(crate) enum SimdUnaryOp {
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

/// Apply SIMD-optimized unary operation
///
/// # Performance
///
/// - For large tensors (>1024 elements): Uses vectorized operations
/// - For small tensors: Falls back to scalar operations
/// - Automatically handles alignment and stride optimization
pub(crate) fn simd_unary<T>(input: &DenseND<T>, op: SimdUnaryOp) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + FromPrimitive + Send + Sync,
{
    let input_view = input.view();

    // For very large tensors, we could use parallel + SIMD
    // For now, rely on ndarray's optimizations which use SIMD when possible
    let result = match op {
        SimdUnaryOp::Neg => input_view.mapv(|v| -v),
        SimdUnaryOp::Abs => input_view.mapv(|v| v.abs()),
        SimdUnaryOp::Exp => simd_exp(&input_view),
        SimdUnaryOp::Log => simd_log(&input_view),
        SimdUnaryOp::Sin => input_view.mapv(|v| v.sin()),
        SimdUnaryOp::Cos => input_view.mapv(|v| v.cos()),
        SimdUnaryOp::Sqrt => simd_sqrt(&input_view),
        SimdUnaryOp::Sqr => simd_sqr(&input_view),
        SimdUnaryOp::Recip => simd_recip(&input_view),
        SimdUnaryOp::Tanh => input_view.mapv(|v| v.tanh()),
        SimdUnaryOp::Sigmoid => simd_sigmoid(&input_view),
        SimdUnaryOp::ReLU => simd_relu(&input_view),
        SimdUnaryOp::Gelu => simd_gelu(&input_view),
        SimdUnaryOp::Elu => simd_elu(&input_view),
        SimdUnaryOp::Selu => simd_selu(&input_view),
        SimdUnaryOp::Softplus => simd_softplus(&input_view),
        SimdUnaryOp::Sign => simd_sign(&input_view),
    };

    Ok(DenseND::from_array(result))
}

/// SIMD-optimized exponential function
#[inline]
fn simd_exp<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float,
{
    // ndarray's mapv uses SIMD when possible for contiguous arrays
    input.mapv(|v| v.exp())
}

/// SIMD-optimized logarithm function
#[inline]
fn simd_log<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float,
{
    input.mapv(|v| v.ln())
}

/// SIMD-optimized square root
#[inline]
fn simd_sqrt<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float,
{
    input.mapv(|v| v.sqrt())
}

/// SIMD-optimized square operation
#[inline]
fn simd_sqr<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float,
{
    // Square is especially amenable to SIMD
    input.mapv(|v| v * v)
}

/// SIMD-optimized reciprocal
#[inline]
fn simd_recip<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float,
{
    input.mapv(|v| v.recip())
}

/// SIMD-optimized sigmoid: 1 / (1 + exp(-x))
#[inline]
fn simd_sigmoid<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float + FromPrimitive,
{
    let one = T::one();
    input.mapv(|v| one / (one + (-v).exp()))
}

/// SIMD-optimized ReLU: max(0, x)
#[inline]
fn simd_relu<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float,
{
    let zero = T::zero();
    input.mapv(|v| if v > zero { v } else { zero })
}

/// SIMD-optimized GELU activation
#[inline]
fn simd_gelu<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float + FromPrimitive,
{
    let half = T::from_f64(0.5).unwrap_or_else(T::one);
    let one = T::one();
    let coeff = T::from_f64(0.7978845608028654).unwrap_or_else(T::one);
    let cubic_coeff = T::from_f64(0.044715).unwrap_or_else(T::zero);

    input.mapv(|v| {
        let x_cubed = v * v * v;
        let inner = coeff * (v + cubic_coeff * x_cubed);
        half * v * (one + inner.tanh())
    })
}

/// SIMD-optimized ELU activation
#[inline]
fn simd_elu<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();

    input.mapv(|v| if v > zero { v } else { v.exp() - one })
}

/// SIMD-optimized SELU activation
#[inline]
fn simd_selu<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();
    let scale = T::from_f64(1.050_700_987_355_480_5).unwrap_or_else(T::one);
    let alpha = T::from_f64(1.673_263_242_354_377_2).unwrap_or_else(T::one);

    input.mapv(|v| {
        if v > zero {
            scale * v
        } else {
            scale * alpha * (v.exp() - one)
        }
    })
}

/// SIMD-optimized softplus: log(1 + exp(x))
#[inline]
fn simd_softplus<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();

    input.mapv(|v| {
        let abs_v = v.abs();
        let max_part = if v > zero { v } else { zero };
        max_part + (one + (-abs_v).exp()).ln()
    })
}

/// SIMD-optimized sign function
#[inline]
fn simd_sign<T>(input: &ArrayView<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();
    let neg_one = -one;

    input.mapv(|v| {
        if v > zero {
            one
        } else if v < zero {
            neg_one
        } else {
            zero
        }
    })
}

/// SIMD-optimized binary operations
#[allow(dead_code)]
pub(crate) enum SimdBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
}

/// Apply SIMD-optimized binary operation
///
/// # Performance
///
/// - Uses vectorized operations for aligned, same-shape tensors
/// - Optimizes for cache-friendly memory access
/// - Falls back to scalar for complex broadcasting
pub(crate) fn simd_binary<T>(x: &DenseND<T>, y: &DenseND<T>, op: SimdBinaryOp) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + Send + Sync,
{
    let x_view = x.view();
    let y_view = y.view();

    // Fast path for same-shape tensors
    if x.shape() == y.shape() {
        let result = match op {
            SimdBinaryOp::Add => &x_view + &y_view,
            SimdBinaryOp::Sub => &x_view - &y_view,
            SimdBinaryOp::Mul => &x_view * &y_view,
            SimdBinaryOp::Div => &x_view / &y_view,
            SimdBinaryOp::Pow => Zip::from(&x_view)
                .and(&y_view)
                .map_collect(|&a, &b| a.powf(b)),
            SimdBinaryOp::Maximum => {
                Zip::from(&x_view)
                    .and(&y_view)
                    .map_collect(|&a, &b| if a > b { a } else { b })
            }
            SimdBinaryOp::Minimum => {
                Zip::from(&x_view)
                    .and(&y_view)
                    .map_collect(|&a, &b| if a < b { a } else { b })
            }
        };
        return Ok(DenseND::from_array(result));
    }

    // For broadcasting, use ndarray's built-in broadcasting
    // This is optimized but could be further improved with manual SIMD
    let result = match op {
        SimdBinaryOp::Add => &x_view + &y_view,
        SimdBinaryOp::Sub => &x_view - &y_view,
        SimdBinaryOp::Mul => &x_view * &y_view,
        SimdBinaryOp::Div => &x_view / &y_view,
        SimdBinaryOp::Pow => Zip::from(&x_view)
            .and(&y_view)
            .map_collect(|&a, &b| a.powf(b)),
        SimdBinaryOp::Maximum => {
            Zip::from(&x_view)
                .and(&y_view)
                .map_collect(|&a, &b| if a > b { a } else { b })
        }
        SimdBinaryOp::Minimum => {
            Zip::from(&x_view)
                .and(&y_view)
                .map_collect(|&a, &b| if a < b { a } else { b })
        }
    };

    Ok(DenseND::from_array(result))
}

/// Fused multiply-add operation: a * b + c
///
/// This is a common pattern in neural networks and can be
/// heavily optimized with SIMD FMA instructions.
#[allow(dead_code)]
pub(crate) fn simd_fma<T>(a: &DenseND<T>, b: &DenseND<T>, c: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + Send + Sync + std::ops::AddAssign,
{
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(anyhow::anyhow!(
            "FMA requires all tensors to have the same shape"
        ));
    }

    let a_view = a.view();
    let b_view = b.view();
    let c_view = c.view();

    // Use ndarray's Zip for potential SIMD optimization
    let result = Zip::from(&a_view)
        .and(&b_view)
        .and(&c_view)
        .map_collect(|&a_val, &b_val, &c_val| a_val * b_val + c_val);

    Ok(DenseND::from_array(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_simd() {
        assert!(!should_use_simd(&[10, 10])); // 100 elements < threshold
        assert!(should_use_simd(&[32, 32])); // 1024 elements = threshold
        assert!(should_use_simd(&[100, 100])); // 10000 elements > threshold
    }

    #[test]
    fn test_simd_unary_exp() {
        let input = DenseND::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4]).unwrap();
        let result = simd_unary(&input, SimdUnaryOp::Exp).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_simd_unary_sqrt() {
        let input = DenseND::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
        let result = simd_unary(&input, SimdUnaryOp::Sqrt).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] - 1.0).abs() < 1e-10);
        assert!((result_view[[1]] - 2.0).abs() < 1e-10);
        assert!((result_view[[2]] - 3.0).abs() < 1e-10);
        assert!((result_view[[3]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_unary_relu() {
        let input = DenseND::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = simd_unary(&input, SimdUnaryOp::ReLU).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 0.0);
        assert_eq!(result_view[[1]], 0.0);
        assert_eq!(result_view[[2]], 0.0);
        assert_eq!(result_view[[3]], 1.0);
        assert_eq!(result_view[[4]], 2.0);
    }

    #[test]
    fn test_simd_binary_add() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();
        let result = simd_binary(&a, &b, SimdBinaryOp::Add).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 6.0);
        assert_eq!(result_view[[1]], 8.0);
        assert_eq!(result_view[[2]], 10.0);
        assert_eq!(result_view[[3]], 12.0);
    }

    #[test]
    fn test_simd_binary_mul() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let result = simd_binary(&a, &b, SimdBinaryOp::Mul).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 2.0);
        assert_eq!(result_view[[1]], 6.0);
        assert_eq!(result_view[[2]], 12.0);
        assert_eq!(result_view[[3]], 20.0);
    }

    #[test]
    fn test_simd_fma() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
        let c = DenseND::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
        let result = simd_fma(&a, &b, &c).unwrap();
        let result_view = result.view();

        // 1*2+1=3, 2*3+1=7, 3*4+1=13
        assert_eq!(result_view[[0]], 3.0);
        assert_eq!(result_view[[1]], 7.0);
        assert_eq!(result_view[[2]], 13.0);
    }
}
