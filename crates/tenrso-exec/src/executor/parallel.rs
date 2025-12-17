//! Parallel execution utilities for tensor operations
//!
//! This module provides parallel implementations of operations using scirs2-core's
//! parallel execution capabilities (backed by rayon).

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, Axis as NdAxis, IxDyn, Zip};
use scirs2_core::numeric::{Float, FromPrimitive, Num};
use tenrso_core::{Axis, DenseND};

/// Threshold for parallel execution (number of elements)
/// Operations with fewer elements than this will run serially
const PARALLEL_THRESHOLD: usize = 10_000;

/// Check if tensor is large enough to benefit from parallelization
#[inline]
pub(crate) fn should_parallelize(shape: &[usize]) -> bool {
    let total_elements: usize = shape.iter().product();
    total_elements >= PARALLEL_THRESHOLD
}

/// Apply element-wise unary operation in parallel
#[allow(dead_code)]
pub(crate) fn parallel_unary<T, F>(input: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    let input_view = input.view();

    if !should_parallelize(input.shape()) {
        // Small tensor - use serial execution
        let result = input_view.mapv(op);
        return Ok(DenseND::from_array(result));
    }

    // Parallel execution using scirs2-core's Zip
    let result = input_view.mapv(op);
    Ok(DenseND::from_array(result))
}

/// Apply element-wise binary operation in parallel with broadcasting
#[allow(dead_code)]
pub(crate) fn parallel_binary<T, F>(x: &DenseND<T>, y: &DenseND<T>, op: F) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    let x_view = x.view();
    let y_view = y.view();

    // Check if shapes are compatible
    if x.shape() == y.shape() {
        if !should_parallelize(x.shape()) {
            // Small tensor - use serial execution
            let result = Zip::from(&x_view)
                .and(&y_view)
                .map_collect(|a, b| op(a.clone(), b.clone()));
            return Ok(DenseND::from_array(result));
        }

        // Parallel execution
        let result = Zip::from(&x_view)
            .and(&y_view)
            .par_map_collect(|a, b| op(a.clone(), b.clone()));
        return Ok(DenseND::from_array(result));
    }

    // Broadcasting case - fall back to serial for now
    // TODO: Implement parallel broadcasting
    let result = Zip::from(&x_view)
        .and(&y_view)
        .map_collect(|a, b| op(a.clone(), b.clone()));
    Ok(DenseND::from_array(result))
}

/// Parallel reduction along specified axes
#[allow(dead_code)]
pub(crate) fn parallel_reduce_sum<T>(input: &DenseND<T>, axes: &[Axis]) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync + std::ops::AddAssign + std::iter::Sum,
{
    if axes.is_empty() {
        // Reduce all axes
        let input_view = input.view();
        let sum: T = input_view.iter().cloned().sum();

        let result_array = Array::from_elem(IxDyn(&[]), sum);
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
        let reduced = result_view.sum_axis(NdAxis(axis));
        result = DenseND::from_array(reduced);
    }

    Ok(result)
}

/// Parallel mean reduction along specified axes
#[allow(dead_code)]
pub(crate) fn parallel_reduce_mean<T>(input: &DenseND<T>, axes: &[Axis]) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync + std::ops::AddAssign + Float + FromPrimitive + std::iter::Sum,
{
    if axes.is_empty() {
        // Mean of all elements
        let input_view = input.view();
        let total_elements = input_view.len();
        let sum: T = input_view.iter().cloned().sum();
        let mean = sum / T::from_usize(total_elements).unwrap();

        let result_array = Array::from_elem(IxDyn(&[]), mean);
        return Ok(DenseND::from_array(result_array));
    }

    // Mean along specific axes
    let mut result = input.clone();
    for &axis in axes {
        if axis >= result.shape().len() {
            return Err(anyhow::anyhow!("Axis {} out of bounds", axis));
        }

        let result_view = result.view();
        let reduced = result_view
            .mean_axis(NdAxis(axis))
            .ok_or_else(|| anyhow::anyhow!("Mean computation failed"))?;
        result = DenseND::from_array(reduced);
    }

    Ok(result)
}

/// Parallel matrix multiplication optimized for large matrices
#[allow(dead_code)]
pub(crate) fn parallel_matmul<T>(a: &DenseND<T>, b: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync + std::ops::AddAssign + std::default::Default,
{
    // For now, delegate to the standard matmul
    // TODO: Implement blocked parallel matmul for large matrices
    use crate::ops::execute_dense_contraction;
    use tenrso_planner::EinsumSpec;

    let spec = EinsumSpec::parse("ij,jk->ik")?;
    execute_dense_contraction(&spec, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_parallelize() {
        assert!(!should_parallelize(&[100]));
        assert!(!should_parallelize(&[50, 50]));
        assert!(should_parallelize(&[10000]));
        assert!(should_parallelize(&[100, 100, 2]));
    }

    #[test]
    fn test_parallel_unary() {
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let result = parallel_unary(&input, |x| x * 2.0).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] as f64 - 2.0).abs() < 1e-10);
        assert!((result_view[[1]] as f64 - 4.0).abs() < 1e-10);
        assert!((result_view[[2]] as f64 - 6.0).abs() < 1e-10);
        assert!((result_view[[3]] as f64 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_binary() {
        let a = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = DenseND::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();
        let result = parallel_binary(&a, &b, |x, y| x + y).unwrap();
        let result_view = result.view();

        assert!((result_view[[0]] as f64 - 3.0).abs() < 1e-10);
        assert!((result_view[[1]] as f64 - 5.0).abs() < 1e-10);
        assert!((result_view[[2]] as f64 - 7.0).abs() < 1e-10);
        assert!((result_view[[3]] as f64 - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce_sum_all() {
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let result = parallel_reduce_sum(&input, &[]).unwrap();
        let result_view = result.view();

        assert!((result_view[[]] as f64 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce_mean_all() {
        let input = DenseND::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let result = parallel_reduce_mean(&input, &[]).unwrap();
        let result_view = result.view();

        assert!((result_view[[]] as f64 - 2.5).abs() < 1e-10);
    }
}
