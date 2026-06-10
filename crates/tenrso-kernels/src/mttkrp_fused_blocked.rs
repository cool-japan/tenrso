//! Fused + cache-blocked MTTKRP combining zero KR materialization with
//! L1-cache-resident result blocks for improved performance.
//!
//! # Algorithm Overview
//!
//! The classic fused MTTKRP (`mttkrp_fused`) loops:
//! ```text
//! for i in 0..mode_size:
//!     for r in 0..cp_rank:
//!         for j in 0..n_cols:
//!             result[i,r] += tensor[i,j] * KR[j,r]   // KR computed on-the-fly
//! ```
//!
//! The **fused + blocked** variant groups `mode_size` rows into cache blocks and
//! computes the KR product for column `j` **once per block** (rather than once
//! per (i, r) pair), then scatters the scaled KR row to all `block_size` result
//! rows in the block. This amortises KR computation by `block_size` and keeps
//! the active result rows resident in L1 cache.
//!
//! # Complexity
//!
//! | Variant | KR multiplications |
//! |---|---|
//! | `mttkrp_fused` | O(mode_size × n_cols × (N-1) × R) |
//! | `mttkrp_fused_blocked` | O((mode_size/B) × n_cols × (N-1) × R) |
//!
//! Where B is `block_size` and N is tensor order.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::{Num, One, Zero};
#[cfg(feature = "parallel")]
use scirs2_core::ndarray_ext::Axis;

use crate::mttkrp::unfold_tensor;

/// Validate MTTKRP inputs: mode, factor count, rank consistency, and row alignment.
/// Returns `(cp_rank, mode_size)` on success.
fn validate_mttkrp_inputs<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<(usize, usize)>
where
    T: Copy + Num,
{
    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    if mode >= rank_tensor {
        anyhow::bail!(
            "Mode {} out of bounds for tensor with rank {}",
            mode,
            rank_tensor
        );
    }
    if factors.len() != rank_tensor {
        anyhow::bail!(
            "Number of factor matrices ({}) must match tensor rank ({})",
            factors.len(),
            rank_tensor
        );
    }
    let cp_rank = factors[0].shape()[1];
    for (i, factor) in factors.iter().enumerate() {
        if factor.shape()[1] != cp_rank {
            anyhow::bail!(
                "Factor matrix {} has {} columns, expected {}",
                i,
                factor.shape()[1],
                cp_rank
            );
        }
        if factor.shape()[0] != tensor_shape[i] {
            anyhow::bail!(
                "Factor matrix {} has {} rows, expected {} (tensor mode-{} size)",
                i,
                factor.shape()[0],
                tensor_shape[i],
                i
            );
        }
    }
    Ok((cp_rank, tensor_shape[mode]))
}

/// Fused + cache-blocked MTTKRP.
///
/// Combines fused KR computation (no materialization of the full KR matrix)
/// with cache-blocking over mode rows. For a block of `block_size` mode rows,
/// the KR product for each non-mode column position is computed once and
/// accumulated into all rows of the block. This amortises KR computation by
/// `block_size` and keeps the result block hot in L1 cache.
///
/// # Arguments
///
/// * `tensor` - Input N-D tensor
/// * `factors` - Factor matrices, one per mode
/// * `mode` - Mode to compute MTTKRP for (0-indexed)
/// * `block_size` - Number of mode rows per cache block (e.g. 32 or 64)
///
/// # Returns
///
/// Matrix of shape `(I_mode, rank)`
///
/// # Errors
///
/// Returns error on shape mismatches (same validation as [`mttkrp`]).
///
/// # Complexity
///
/// Time: O(mode_size × rank × n_cols + (mode_size/block_size) × n_cols × (N-1) × rank)
/// Space: O(mode_size × rank) output + O(rank) kr_row buffer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_blocked;
///
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f64).collect()
/// ).unwrap();
/// let u1 = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
/// let u2 = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();
/// let u3 = Array2::from_shape_vec((6, 3), vec![1.0; 18]).unwrap();
///
/// let result = mttkrp_fused_blocked(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
///     32,
/// ).unwrap();
/// assert_eq!(result.shape(), &[5, 3]);
/// ```
pub fn mttkrp_fused_blocked<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
    block_size: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + 'static,
{
    let (cp_rank, mode_size) = validate_mttkrp_inputs(tensor, factors, mode)?;

    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    let unfolded = unfold_tensor(tensor, mode)?;
    let n_cols = unfolded.shape()[1];

    // Other dimensions (all except `mode`) in their natural (unfold) order
    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&d| d != mode).collect();

    // Strides for decoding column index j to per-dimension indices (FORWARD order)
    let mut strides = vec![1usize; other_dims.len()];
    for idx in (0..other_dims.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1] * tensor_shape[other_dims[idx + 1]];
    }

    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    // Reusable KR row buffer — shared across all rows in a block for a given j
    let mut kr_row = vec![T::one(); cp_rank];

    // Effective block size: at least 1 so we never divide by zero
    let effective_block = block_size.max(1);

    let mut block_start = 0;
    while block_start < mode_size {
        let block_end = (block_start + effective_block).min(mode_size);

        // For each unfolded column j:
        //   1. Compute KR[j, :] once for this block (all ranks simultaneously)
        //   2. For each mode row i in [block_start, block_end):
        //         result[i, r] += unfolded[i, j] * kr_row[r]
        for j in 0..n_cols {
            // --- Build kr_row[r] = product over all non-mode dims ---
            kr_row.iter_mut().for_each(|v| *v = T::one());
            for (idx, &dim) in other_dims.iter().enumerate() {
                let dim_size = tensor_shape[dim];
                let dim_idx = (j / strides[idx]) % dim_size;
                let factor_row = factors[dim].row(dim_idx);
                for r in 0..cp_rank {
                    kr_row[r] = kr_row[r] * factor_row[r];
                }
            }

            // --- Scatter to all mode rows in this block ---
            for i in block_start..block_end {
                let x_val = unfolded[[i, j]];
                for r in 0..cp_rank {
                    result[[i, r]] = result[[i, r]] + x_val * kr_row[r];
                }
            }
        }

        block_start = block_end;
    }

    Ok(result)
}

/// Parallel fused + cache-blocked MTTKRP via Rayon.
///
/// Partitions mode-row blocks across Rayon threads using
/// `axis_chunks_iter_mut`. Each thread owns a disjoint set of result rows
/// and a local `kr_row` buffer; no atomics or locks are needed.
///
/// # Arguments
///
/// * `tensor` - Input N-D tensor
/// * `factors` - Factor matrices, one per mode
/// * `mode` - Mode to compute MTTKRP for (0-indexed)
/// * `block_size` - Number of mode rows per cache block (also the parallel grain)
///
/// # Returns
///
/// Matrix of shape `(I_mode, rank)`
///
/// # Errors
///
/// Returns error on shape mismatches (same validation as [`mttkrp`]).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_blocked_parallel;
///
/// let tensor = Array::from_shape_vec(
///     vec![8, 9, 10],
///     (0..720).map(|x| x as f64).collect()
/// ).unwrap();
/// let u1 = Array2::from_shape_vec((8, 4), vec![1.0; 32]).unwrap();
/// let u2 = Array2::from_shape_vec((9, 4), vec![1.0; 36]).unwrap();
/// let u3 = Array2::from_shape_vec((10, 4), vec![1.0; 40]).unwrap();
///
/// let result = mttkrp_fused_blocked_parallel(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
///     32,
/// ).unwrap();
/// assert_eq!(result.shape(), &[9, 4]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_fused_blocked_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
    block_size: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + 'static + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let (cp_rank, mode_size) = validate_mttkrp_inputs(tensor, factors, mode)?;

    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    let unfolded = unfold_tensor(tensor, mode)?;
    let n_cols = unfolded.shape()[1];

    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&d| d != mode).collect();

    let mut strides = vec![1usize; other_dims.len()];
    for idx in (0..other_dims.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1] * tensor_shape[other_dims[idx + 1]];
    }

    let effective_block = block_size.max(1);
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    // Each parallel chunk owns a disjoint slice of result rows.
    // `axis_chunks_iter_mut(Axis(0), effective_block)` splits result into
    // non-overlapping row-block views; each view is sent to one thread.
    result
        .axis_chunks_iter_mut(Axis(0), effective_block)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut result_chunk)| {
            let block_start = chunk_idx * effective_block;
            let block_rows = result_chunk.shape()[0]; // actual rows in this chunk

            // Per-thread KR buffer — allocated once per task
            let mut kr_row = vec![T::one(); cp_rank];

            for j in 0..n_cols {
                // Build KR product for column j
                kr_row.iter_mut().for_each(|v| *v = T::one());
                for (idx, &dim) in other_dims.iter().enumerate() {
                    let dim_size = tensor_shape[dim];
                    let dim_idx = (j / strides[idx]) % dim_size;
                    let factor_row = factors[dim].row(dim_idx);
                    for r in 0..cp_rank {
                        kr_row[r] = kr_row[r] * factor_row[r];
                    }
                }

                // Scatter to all rows in this chunk
                for i_local in 0..block_rows {
                    let x_val = unfolded[[block_start + i_local, j]];
                    for r in 0..cp_rank {
                        result_chunk[[i_local, r]] =
                            result_chunk[[i_local, r]] + x_val * kr_row[r];
                    }
                }
            }
        });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mttkrp::mttkrp;
    use scirs2_core::ndarray_ext::{array, Array};

    /// Helper: max absolute difference between two Array2<f64>
    fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    // ─── Test 1: 4×5×6 mode 0 ────────────────────────────────────────────

    #[test]
    fn test_fused_blocked_matches_mttkrp_mode0() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let u1 = array![[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9], [1.0, 0.0, 0.5]];
        let u2 = Array::from_shape_vec(
            (5, 3),
            (0..15).map(|x| (x as f64) * 0.1).collect(),
        )
        .unwrap();
        let u3 = Array::from_shape_vec(
            (6, 3),
            (0..18).map(|x| (x as f64) * 0.05).collect(),
        )
        .unwrap();

        let expected = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 0).unwrap();
        let blocked =
            mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 0, 2)
                .unwrap();

        assert_eq!(expected.shape(), blocked.shape());
        assert!(
            max_diff(&expected, &blocked) < 1e-10,
            "mode 0 max diff = {}",
            max_diff(&expected, &blocked)
        );
    }

    // ─── Test 2: 4×5×6 mode 1 ────────────────────────────────────────────

    #[test]
    fn test_fused_blocked_matches_mttkrp_mode1() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let u1 = Array::from_shape_vec((4, 3), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();
        let u2 = Array::from_shape_vec((5, 3), (0..15).map(|x| x as f64 * 0.2).collect()).unwrap();
        let u3 = Array::from_shape_vec((6, 3), (0..18).map(|x| x as f64 * 0.15).collect()).unwrap();

        let expected = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let blocked =
            mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1, 2)
                .unwrap();

        assert_eq!(expected.shape(), blocked.shape());
        assert!(
            max_diff(&expected, &blocked) < 1e-10,
            "mode 1 max diff = {}",
            max_diff(&expected, &blocked)
        );
    }

    // ─── Test 3: 4×5×6 mode 2 ────────────────────────────────────────────

    #[test]
    fn test_fused_blocked_matches_mttkrp_mode2() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let u1 = Array::from_shape_vec((4, 3), (0..12).map(|x| x as f64 * 0.3).collect()).unwrap();
        let u2 = Array::from_shape_vec((5, 3), (0..15).map(|x| x as f64 * 0.4).collect()).unwrap();
        let u3 = Array::from_shape_vec((6, 3), (0..18).map(|x| x as f64 * 0.25).collect()).unwrap();

        let expected = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 2).unwrap();
        let blocked =
            mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 2, 2)
                .unwrap();

        assert_eq!(expected.shape(), blocked.shape());
        assert!(
            max_diff(&expected, &blocked) < 1e-10,
            "mode 2 max diff = {}",
            max_diff(&expected, &blocked)
        );
    }

    // ─── Test 4: block_size=1 (each row is its own block) ─────────────────

    #[test]
    fn test_fused_blocked_block_size_1() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let u1 = array![[1.0, 0.5], [0.5, 1.0], [0.8, 0.2], [0.3, 0.7]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3]];
        let u3 = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.6, 0.4],
            [0.1, 0.9]
        ];

        let expected = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let blocked =
            mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1, 1)
                .unwrap();

        assert_eq!(expected.shape(), blocked.shape());
        assert!(
            max_diff(&expected, &blocked) < 1e-10,
            "block_size=1 max diff = {}",
            max_diff(&expected, &blocked)
        );
    }

    // ─── Test 5: block_size larger than mode_size ─────────────────────────

    #[test]
    fn test_fused_blocked_block_size_large() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let u1 = Array::from_shape_vec((4, 3), vec![0.5; 12]).unwrap();
        let u2 = Array::from_shape_vec((5, 3), vec![0.5; 15]).unwrap();
        let u3 = Array::from_shape_vec((6, 3), vec![0.5; 18]).unwrap();

        // block_size = 9999 >> mode_size = 5
        let expected = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let blocked =
            mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1, 9999)
                .unwrap();

        assert_eq!(expected.shape(), blocked.shape());
        assert!(
            max_diff(&expected, &blocked) < 1e-10,
            "large block_size max diff = {}",
            max_diff(&expected, &blocked)
        );
    }

    // ─── Test 6: 4-D tensor ───────────────────────────────────────────────

    #[test]
    fn test_fused_blocked_4d() {
        let tensor =
            Array::from_shape_vec(vec![3, 4, 5, 6], (0..360).map(|x| x as f64).collect())
                .unwrap();
        let u0 = Array::from_shape_vec((3, 4), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();
        let u1 = Array::from_shape_vec((4, 4), (0..16).map(|x| x as f64 * 0.15).collect()).unwrap();
        let u2 = Array::from_shape_vec((5, 4), (0..20).map(|x| x as f64 * 0.2).collect()).unwrap();
        let u3 = Array::from_shape_vec((6, 4), (0..24).map(|x| x as f64 * 0.25).collect()).unwrap();

        let expected =
            mttkrp(&tensor.view(), &[u0.view(), u1.view(), u2.view(), u3.view()], 1).unwrap();
        let blocked = mttkrp_fused_blocked(
            &tensor.view(),
            &[u0.view(), u1.view(), u2.view(), u3.view()],
            1,
            2,
        )
        .unwrap();

        assert_eq!(expected.shape(), blocked.shape());
        assert!(
            max_diff(&expected, &blocked) < 1e-10,
            "4D max diff = {}",
            max_diff(&expected, &blocked)
        );
    }

    // ─── Test 7: invalid mode error propagation ────────────────────────────

    #[test]
    #[should_panic(expected = "Mode")]
    fn test_fused_blocked_invalid_mode() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [1.0], [1.0]];

        mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view()], 5, 4).unwrap();
    }

    // ─── Test 8: wrong number of factors ──────────────────────────────────

    #[test]
    #[should_panic(expected = "Number of factor matrices")]
    fn test_fused_blocked_wrong_num_factors() {
        let tensor = Array::from_shape_vec(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [1.0], [1.0]];

        // Only 2 factors for a 3D tensor
        mttkrp_fused_blocked(&tensor.view(), &[u1.view(), u2.view()], 0, 4).unwrap();
    }

    // ─── Test 9 (parallel): parallel matches serial ───────────────────────

    #[cfg(feature = "parallel")]
    #[test]
    fn test_fused_blocked_parallel_matches_serial() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();
        let u1 = Array::from_shape_vec((4, 3), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();
        let u2 = Array::from_shape_vec((5, 3), (0..15).map(|x| x as f64 * 0.2).collect()).unwrap();
        let u3 = Array::from_shape_vec((6, 3), (0..18).map(|x| x as f64 * 0.15).collect()).unwrap();

        // Test all three modes
        for mode in 0..3 {
            let serial = mttkrp_fused_blocked(
                &tensor.view(),
                &[u1.view(), u2.view(), u3.view()],
                mode,
                2,
            )
            .unwrap();
            let par = mttkrp_fused_blocked_parallel(
                &tensor.view(),
                &[u1.view(), u2.view(), u3.view()],
                mode,
                2,
            )
            .unwrap();

            assert_eq!(serial.shape(), par.shape(), "mode {} shape mismatch", mode);
            assert!(
                max_diff(&serial, &par) < 1e-10,
                "mode {} parallel vs serial max diff = {}",
                mode,
                max_diff(&serial, &par)
            );
        }
    }
}
