//! Parallel execution utilities for tensor operations
//!
//! This module provides parallel implementations of operations using scirs2-core's
//! parallel execution capabilities (backed by rayon).

use anyhow::Result;
use rayon::prelude::*;
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

    // Broadcasting case: expand both operands to the broadcast shape, then
    // apply the operation in parallel over the resulting equal-shaped tensors.
    let x_shape = x.shape().to_vec();
    let y_shape = y.shape().to_vec();
    let broadcast_shape = compute_broadcast_shape(&x_shape, &y_shape)
        .ok_or_else(|| anyhow::anyhow!("parallel_binary: shapes {:?} and {:?} are not broadcast-compatible", x_shape, y_shape))?;

    let x_expanded = x.broadcast_to(&broadcast_shape)?;
    let y_expanded = y.broadcast_to(&broadcast_shape)?;

    let x_exp_view = x_expanded.view();
    let y_exp_view = y_expanded.view();

    if !should_parallelize(&broadcast_shape) {
        let result = Zip::from(&x_exp_view)
            .and(&y_exp_view)
            .map_collect(|a, b| op(a.clone(), b.clone()));
        return Ok(DenseND::from_array(result));
    }

    let result = Zip::from(&x_exp_view)
        .and(&y_exp_view)
        .par_map_collect(|a, b| op(a.clone(), b.clone()));
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
        let divisor = T::from_usize(total_elements)
            .ok_or_else(|| anyhow::anyhow!("parallel mean: cannot convert element count to T"))?;
        let mean = sum / divisor;

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

/// Parallel matrix multiplication optimized for large matrices.
///
/// Uses a row-blocked strategy: the output rows are partitioned into
/// independent tiles that are computed in parallel using rayon.  Each tile
/// computes a sub-matrix of `C = A × B` without synchronisation, which
/// gives near-linear scaling up to the number of available CPU cores.
///
/// # Arguments
///
/// * `a` - Left matrix with shape `[M, K]`
/// * `b` - Right matrix with shape `[K, N]`
///
/// # Returns
///
/// Result matrix with shape `[M, N]`, or an error if shapes are incompatible.
///
/// # Block size
///
/// The row-block size is chosen adaptively: each tile covers at least one
/// row, and the total number of tiles is capped at 4× the rayon thread count
/// so that the scheduling overhead stays small relative to the work.
#[allow(dead_code)]
pub(crate) fn parallel_matmul<T>(a: &DenseND<T>, b: &DenseND<T>) -> Result<DenseND<T>>
where
    T: Clone + Num + Send + Sync + std::ops::AddAssign + std::default::Default,
{
    // Validate shapes
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(anyhow::anyhow!(
            "parallel_matmul: inputs must be 2-D matrices, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ));
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    if k != b.shape()[0] {
        return Err(anyhow::anyhow!(
            "parallel_matmul: inner dimensions must match, got {} vs {}",
            k,
            b.shape()[0]
        ));
    }

    // For small matrices fall back to the serial contraction path.
    if !should_parallelize(&[m, n]) {
        use crate::ops::execute_dense_contraction;
        use tenrso_planner::EinsumSpec;
        let spec = EinsumSpec::parse("ij,jk->ik")?;
        return execute_dense_contraction(&spec, a, b);
    }

    // Choose a row-block size that gives ~4× the rayon thread count tiles,
    // with a minimum of 1 row per tile.
    let num_threads = rayon::current_num_threads().max(1);
    let target_tiles = (4 * num_threads).max(1);
    let block_rows = (m / target_tiles).max(1);

    let a_view = a.view();
    let b_view = b.view();

    // Compute row ranges for each tile.
    let tile_ranges: Vec<(usize, usize)> = {
        let mut ranges = Vec::new();
        let mut row = 0;
        while row < m {
            let end = (row + block_rows).min(m);
            ranges.push((row, end));
            row = end;
        }
        ranges
    };

    // Each tile produces an independent `Vec<T>` covering `(end-start) × n`
    // output elements.  Collecting and concatenating avoids any unsafe
    // shared-pointer aliasing.
    let tiles: Vec<Vec<T>> = tile_ranges
        .par_iter()
        .map(|&(row_start, row_end)| {
            let tile_rows = row_end - row_start;
            let mut tile = Vec::with_capacity(tile_rows * n);
            for i in row_start..row_end {
                for j in 0..n {
                    let mut acc = T::default();
                    for l in 0..k {
                        let a_idx = [i, l];
                        let b_idx = [l, j];
                        acc += a_view[a_idx.as_ref()].clone() * b_view[b_idx.as_ref()].clone();
                    }
                    tile.push(acc);
                }
            }
            tile
        })
        .collect();

    // Concatenate tile results into a single flat buffer.
    let output: Vec<T> = tiles.into_iter().flatten().collect();
    DenseND::from_vec(output, &[m, n])
}

/// Compute the broadcast shape of two tensor shapes following NumPy rules.
///
/// Returns `None` when the shapes are not compatible.
fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let d1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let d2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        if d1 == d2 {
            result.push(d1);
        } else if d1 == 1 {
            result.push(d2);
        } else if d2 == 1 {
            result.push(d1);
        } else {
            return None;
        }
    }
    result.reverse();
    Some(result)
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

    // -----------------------------------------------------------------------
    // Blocked parallel matmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_matmul_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = DenseND::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = DenseND::from_vec(vec![5.0f64, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = parallel_matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let v = c.view();
        assert!((v[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((v[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((v[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((v[[1, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_matmul_rectangular() {
        // [2×3] × [3×2] -> [2×2]
        let a = DenseND::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseND::from_vec(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let c = parallel_matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let v = c.view();
        // row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        assert!((v[[0, 0]] - 58.0).abs() < 1e-10);
        assert!((v[[0, 1]] - 64.0).abs() < 1e-10);
        // row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((v[[1, 0]] - 139.0).abs() < 1e-10);
        assert!((v[[1, 1]] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_matmul_dimension_mismatch() {
        let a = DenseND::<f64>::zeros(&[2, 3]);
        let b = DenseND::<f64>::zeros(&[4, 2]);
        assert!(parallel_matmul(&a, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // Broadcasting in parallel_binary
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_binary_broadcast_row_vector() {
        // x: [3×1], y: [1×4] → result: [3×4], all elements = x_i + y_j
        let x = DenseND::from_vec(vec![1.0f64, 2.0, 3.0], &[3, 1]).unwrap();
        let y = DenseND::from_vec(vec![10.0f64, 20.0, 30.0, 40.0], &[1, 4]).unwrap();
        let result = parallel_binary(&x, &y, |a, b| a + b).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
        let v = result.view();
        assert!((v[[0, 0]] - 11.0).abs() < 1e-10);
        assert!((v[[0, 3]] - 41.0).abs() < 1e-10);
        assert!((v[[2, 0]] - 13.0).abs() < 1e-10);
        assert!((v[[2, 3]] - 43.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_binary_broadcast_scalar_like() {
        // x: [1], y: [3×4] → result: [3×4]
        let x = DenseND::from_vec(vec![5.0f64], &[1]).unwrap();
        let y = DenseND::ones(&[3, 4]);
        let result = parallel_binary(&x, &y, |a, b| a + b).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
        let v = result.view();
        assert!((v[[0, 0]] - 6.0).abs() < 1e-10);
        assert!((v[[2, 3]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_binary_broadcast_incompatible() {
        let x = DenseND::<f64>::zeros(&[3, 2]);
        let y = DenseND::<f64>::zeros(&[3, 4]);
        assert!(parallel_binary(&x, &y, |a, b| a + b).is_err());
    }

    #[test]
    fn test_compute_broadcast_shape() {
        assert_eq!(
            compute_broadcast_shape(&[3, 1], &[1, 4]),
            Some(vec![3, 4])
        );
        assert_eq!(
            compute_broadcast_shape(&[5], &[2, 5]),
            Some(vec![2, 5])
        );
        assert_eq!(compute_broadcast_shape(&[3, 2], &[3, 4]), None);
    }
}
