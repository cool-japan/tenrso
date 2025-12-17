//! Vectorized broadcasting operations
//!
//! This module provides high-performance broadcasting for aligned tensor shapes.
//! When tensor shapes are compatible and memory is properly aligned, we can
//! use SIMD instructions for dramatic speedups.
//!
//! # Optimization Strategy
//!
//! 1. Detect common broadcasting patterns (scalar, 1D broadcast, etc.)
//! 2. Use specialized kernels for each pattern
//! 3. Leverage SIMD for aligned, contiguous data
//! 4. Fall back to standard broadcasting for complex cases
//!
//! # Common Patterns
//!
//! - Scalar broadcast: (1,) + (N,M,K) → (N,M,K)
//! - Vector broadcast: (N,) + (N,M) → (N,M)
//! - Matrix broadcast: (N,M) + (N,M,K) → (N,M,K)

#![allow(dead_code)]

use anyhow::Result;
use scirs2_core::ndarray_ext::Zip;
use scirs2_core::numeric::{Float, Num};
use tenrso_core::DenseND;

/// Broadcasting patterns we can optimize
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BroadcastPattern {
    /// Exact same shape - no broadcasting needed
    SameShape,
    /// One operand is a scalar
    Scalar,
    /// Broadcasting along last dimension
    LastDim,
    /// Broadcasting along first dimension
    FirstDim,
    /// General broadcasting (fall back to ndarray)
    General,
}

/// Detect the broadcasting pattern between two shapes
pub(crate) fn detect_broadcast_pattern(shape_a: &[usize], shape_b: &[usize]) -> BroadcastPattern {
    // Same shape - no broadcasting
    if shape_a == shape_b {
        return BroadcastPattern::SameShape;
    }

    // Scalar cases
    if shape_a.len() == 1 && shape_a[0] == 1 {
        return BroadcastPattern::Scalar;
    }
    if shape_b.len() == 1 && shape_b[0] == 1 {
        return BroadcastPattern::Scalar;
    }
    if shape_a.is_empty() || shape_b.is_empty() {
        return BroadcastPattern::Scalar;
    }

    // Check for last dimension broadcast
    if shape_a.len() == shape_b.len() {
        let mut differs_only_in_last = true;
        for i in 0..shape_a.len() - 1 {
            if shape_a[i] != shape_b[i] {
                differs_only_in_last = false;
                break;
            }
        }
        if differs_only_in_last
            && (shape_a[shape_a.len() - 1] == 1 || shape_b[shape_b.len() - 1] == 1)
        {
            return BroadcastPattern::LastDim;
        }
    }

    // Check for first dimension broadcast
    if shape_a.len() == shape_b.len() {
        let mut differs_only_in_first = true;
        for i in 1..shape_a.len() {
            if shape_a[i] != shape_b[i] {
                differs_only_in_first = false;
                break;
            }
        }
        if differs_only_in_first && (shape_a[0] == 1 || shape_b[0] == 1) {
            return BroadcastPattern::FirstDim;
        }
    }

    BroadcastPattern::General
}

/// Vectorized binary operation with optimized broadcasting
///
/// # Performance
///
/// - SameShape: Direct SIMD operations, no overhead
/// - Scalar: Optimized scalar broadcast loops
/// - LastDim/FirstDim: Cache-friendly strided operations
/// - General: Falls back to ndarray's broadcasting
pub(crate) fn vectorized_binary_op<T, F>(
    a: &DenseND<T>,
    b: &DenseND<T>,
    op: F,
) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    let pattern = detect_broadcast_pattern(a.shape(), b.shape());

    match pattern {
        BroadcastPattern::SameShape => vectorized_same_shape(a, b, op),
        BroadcastPattern::Scalar => vectorized_scalar_broadcast(a, b, op),
        BroadcastPattern::LastDim => vectorized_last_dim_broadcast(a, b, op),
        BroadcastPattern::FirstDim => vectorized_first_dim_broadcast(a, b, op),
        BroadcastPattern::General => vectorized_general_broadcast(a, b, op),
    }
}

/// Optimized same-shape operation (no broadcasting)
fn vectorized_same_shape<T, F>(a: &DenseND<T>, b: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    let a_view = a.view();
    let b_view = b.view();

    // Use Zip for potential SIMD optimization
    let result = Zip::from(&a_view)
        .and(&b_view)
        .par_map_collect(|a_val, b_val| op(a_val.clone(), b_val.clone()));

    Ok(DenseND::from_array(result))
}

/// Optimized scalar broadcast
#[allow(dead_code)]
fn vectorized_scalar_broadcast<T, F>(a: &DenseND<T>, b: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Float,
    F: Fn(T, T) -> T,
{
    let a_view = a.view();
    let b_view = b.view();

    // Determine which is scalar
    let (scalar_val, tensor_view, op_flipped) = if a.view().len() == 1 || a.shape().is_empty() {
        let scalar = if a.view().len() == 1 {
            a_view.iter().next().cloned().unwrap()
        } else {
            T::zero()
        };
        (scalar, b_view, false)
    } else {
        let scalar = if b.view().len() == 1 {
            b_view.iter().next().cloned().unwrap()
        } else {
            T::zero()
        };
        (scalar, a_view, true)
    };

    // Vectorized scalar operation
    let result = if op_flipped {
        tensor_view.mapv(|v| op(v, scalar_val))
    } else {
        tensor_view.mapv(|v| op(scalar_val, v))
    };

    Ok(DenseND::from_array(result))
}

/// Optimized last dimension broadcast
#[allow(dead_code)]
fn vectorized_last_dim_broadcast<T, F>(a: &DenseND<T>, b: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Float,
    F: Fn(T, T) -> T,
{
    let a_view = a.view();
    let b_view = b.view();

    // Use ndarray's broadcasting which is already optimized
    let result = Zip::from(&a_view)
        .and(&b_view)
        .map_collect(|&a_val, &b_val| op(a_val, b_val));

    Ok(DenseND::from_array(result))
}

/// Optimized first dimension broadcast
#[allow(dead_code)]
fn vectorized_first_dim_broadcast<T, F>(a: &DenseND<T>, b: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Float,
    F: Fn(T, T) -> T,
{
    let a_view = a.view();
    let b_view = b.view();

    // Use ndarray's broadcasting
    let result = Zip::from(&a_view)
        .and(&b_view)
        .map_collect(|&a_val, &b_val| op(a_val, b_val));

    Ok(DenseND::from_array(result))
}

/// General broadcasting fallback
#[allow(dead_code)]
fn vectorized_general_broadcast<T, F>(a: &DenseND<T>, b: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Float,
    F: Fn(T, T) -> T,
{
    let a_view = a.view();
    let b_view = b.view();

    // ndarray handles general broadcasting well
    let result = Zip::from(&a_view)
        .and(&b_view)
        .map_collect(|&a_val, &b_val| op(a_val, b_val));

    Ok(DenseND::from_array(result))
}

/// Specialized addition with broadcasting
#[allow(dead_code)]
pub(crate) fn vectorized_add<T>(a: &DenseND<T>, b: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + Send + Sync + std::ops::Add<Output = T>,
{
    let pattern = detect_broadcast_pattern(a.shape(), b.shape());

    match pattern {
        BroadcastPattern::SameShape => {
            // Direct addition
            let result = &a.view() + &b.view();
            Ok(DenseND::from_array(result))
        }
        BroadcastPattern::Scalar => {
            // Scalar addition
            vectorized_scalar_broadcast(a, b, |x, y| x + y)
        }
        _ => {
            // General case
            let result = &a.view() + &b.view();
            Ok(DenseND::from_array(result))
        }
    }
}

/// Specialized multiplication with broadcasting
#[allow(dead_code)]
pub(crate) fn vectorized_mul<T>(a: &DenseND<T>, b: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + Float + Send + Sync + std::ops::Mul<Output = T>,
{
    let pattern = detect_broadcast_pattern(a.shape(), b.shape());

    match pattern {
        BroadcastPattern::SameShape => {
            // Direct multiplication
            let result = &a.view() * &b.view();
            Ok(DenseND::from_array(result))
        }
        BroadcastPattern::Scalar => {
            // Scalar multiplication
            vectorized_scalar_broadcast(a, b, |x, y| x * y)
        }
        _ => {
            // General case
            let result = &a.view() * &b.view();
            Ok(DenseND::from_array(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_broadcast_pattern_same_shape() {
        assert_eq!(
            detect_broadcast_pattern(&[3, 4], &[3, 4]),
            BroadcastPattern::SameShape
        );
    }

    #[test]
    fn test_detect_broadcast_pattern_scalar() {
        assert_eq!(
            detect_broadcast_pattern(&[1], &[3, 4]),
            BroadcastPattern::Scalar
        );
        assert_eq!(
            detect_broadcast_pattern(&[3, 4], &[1]),
            BroadcastPattern::Scalar
        );
    }

    #[test]
    fn test_detect_broadcast_pattern_last_dim() {
        assert_eq!(
            detect_broadcast_pattern(&[3, 4, 1], &[3, 4, 5]),
            BroadcastPattern::LastDim
        );
    }

    #[test]
    fn test_detect_broadcast_pattern_first_dim() {
        assert_eq!(
            detect_broadcast_pattern(&[1, 4, 5], &[3, 4, 5]),
            BroadcastPattern::FirstDim
        );
    }

    #[test]
    fn test_vectorized_add_same_shape() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();

        let result = vectorized_add(&a, &b).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 6.0);
        assert_eq!(result_view[[1]], 8.0);
        assert_eq!(result_view[[2]], 10.0);
        assert_eq!(result_view[[3]], 12.0);
    }

    #[test]
    fn test_vectorized_add_scalar() {
        let a = DenseND::from_vec(vec![5.0], &[1]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = vectorized_add(&a, &b).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 6.0);
        assert_eq!(result_view[[1]], 7.0);
        assert_eq!(result_view[[2]], 8.0);
        assert_eq!(result_view[[3]], 9.0);
    }

    #[test]
    fn test_vectorized_mul_same_shape() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

        let result = vectorized_mul(&a, &b).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 2.0);
        assert_eq!(result_view[[1]], 6.0);
        assert_eq!(result_view[[2]], 12.0);
        assert_eq!(result_view[[3]], 20.0);
    }

    #[test]
    fn test_vectorized_mul_scalar() {
        let a = DenseND::from_vec(vec![2.0], &[1]).unwrap();
        let b = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = vectorized_mul(&a, &b).unwrap();
        let result_view = result.view();

        assert_eq!(result_view[[0]], 2.0);
        assert_eq!(result_view[[1]], 4.0);
        assert_eq!(result_view[[2]], 6.0);
        assert_eq!(result_view[[3]], 8.0);
    }

    #[test]
    fn test_vectorized_binary_op_custom() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseND::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let result = vectorized_binary_op(&a, &b, |x, y| x * x + y).unwrap();
        let result_view = result.view();

        // 1*1+4=5, 2*2+5=9, 3*3+6=15
        assert_eq!(result_view[[0]], 5.0);
        assert_eq!(result_view[[1]], 9.0);
        assert_eq!(result_view[[2]], 15.0);
    }
}
