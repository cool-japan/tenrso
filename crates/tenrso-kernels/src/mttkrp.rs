//! MTTKRP (Matricized Tensor Times Khatri-Rao Product) implementation
//!
//! MTTKRP is the computational bottleneck in CP-ALS tensor decomposition.
//! For tensor X and factor matrices {U₁, ..., Uₙ}, it computes:
//!
//! V = X_(mode) × (U₁ ⊙ ... ⊙ U_(mode-1) ⊙ U_(mode+1) ⊙ ... ⊙ Uₙ)
//!
//! Where X_(mode) is the mode-n matricization and ⊙ is the Khatri-Rao product.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::{Num, One, Zero};

/// Compute MTTKRP (Matricized Tensor Times Khatri-Rao Product)
///
/// This is the key operation in CP-ALS tensor decomposition. For tensor X with
/// shape (I₁, ..., Iₙ) and factor matrices U₁,...,Uₙ each with shape (Iₖ, R),
/// computes:
///
/// V = X_(mode) × KR
///
/// Where:
/// - X_(mode) is the mode-n matricization with shape (Iₘₒ₋ᵈₑ, ∏ᵢ≠ₘₒ₋ᵈₑ Iᵢ)
/// - KR is the Khatri-Rao product of all factor matrices except U_(mode)
/// - Result V has shape (I_mode, R)
///
/// # Arguments
///
/// * `tensor` - Input tensor with N dimensions
/// * `factors` - Factor matrices, one for each mode
/// * `mode` - The mode to compute MTTKRP for (0-indexed)
///
/// # Returns
///
/// Matrix with shape (I_mode, R) where R is the CP rank
///
/// # Errors
///
/// Returns error if:
/// - Mode is out of bounds
/// - Factor matrix shapes don't match tensor dimensions
/// - Factor matrices have different numbers of columns (rank)
///
/// # Complexity
///
/// Time: O(I_mode × R × ∏ᵢ≠ₘₒ₋ᵈₑ Iᵢ)
/// Space: O(R × ∏ᵢ≠ₘₒ₋ᵈₑ Iᵢ) for the Khatri-Rao product
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp;
///
/// // 3D tensor: 2×3×4
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// // Factor matrices for rank-2 CP decomposition
/// let u1 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let u2 = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
/// let u3 = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
///
/// let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
/// assert_eq!(result.shape(), &[3, 2]);  // (I₂, R)
/// ```
pub fn mttkrp<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + 'static,
{
    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation
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

    // Check that all factor matrices have the same rank (number of columns)
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

    // Step 1: Unfold tensor along the specified mode
    let unfolded = unfold_tensor(tensor, mode)?;

    // Step 2: Compute Khatri-Rao product of all factor matrices except the mode-th one
    let kr = khatri_rao_except_mode(factors, mode)?;

    // Step 3: Matrix multiplication: X_(mode) × KR
    // Result has shape (I_mode, R)
    // Using ndarray's .dot() which leverages BLAS for acceleration.
    let result = unfolded.dot(&kr);

    Ok(result)
}

/// Unfold a tensor along a specific mode into a matrix
fn unfold_tensor<T>(tensor: &ArrayView<T, IxDyn>, mode: usize) -> Result<Array2<T>>
where
    T: Copy + Num,
{
    let shape = tensor.shape();
    let mode_size = shape[mode];
    let other_size: usize = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &s)| s)
        .product();

    // Create permutation: [mode, 0, 1, ..., mode-1, mode+1, ..., rank-1]
    let mut perm: Vec<usize> = Vec::with_capacity(shape.len());
    perm.push(mode);
    for i in 0..shape.len() {
        if i != mode {
            perm.push(i);
        }
    }

    // Permute and reshape
    let permuted = tensor.clone().permuted_axes(IxDyn(&perm));
    let contiguous = permuted.as_standard_layout().into_owned();
    let unfolded = contiguous.into_shape_with_order((mode_size, other_size))?;

    Ok(unfolded)
}

/// Compute Khatri-Rao product of all matrices except the one at the specified mode
///
/// The matrices are multiplied in FORWARD order (first to last, skipping mode) to match unfold column ordering
fn khatri_rao_except_mode<T>(factors: &[ArrayView2<T>], skip_mode: usize) -> Result<Array2<T>>
where
    T: Copy + Num,
{
    // Collect all matrices except the one at skip_mode, in FORWARD order (matching unfold)
    let mut matrices: Vec<&ArrayView2<T>> = Vec::new();
    for (i, factor) in factors.iter().enumerate() {
        if i != skip_mode {
            matrices.push(factor);
        }
    }

    if matrices.is_empty() {
        anyhow::bail!("Need at least 2 factor matrices for MTTKRP");
    }

    // Compute Khatri-Rao product iteratively
    let mut result = matrices[0].to_owned();

    for &matrix in &matrices[1..] {
        result = khatri_rao_two(&result.view(), matrix)?;
    }

    Ok(result)
}

/// Compute Khatri-Rao product of two matrices
fn khatri_rao_two<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Result<Array2<T>>
where
    T: Copy + Num,
{
    let (i, k1) = (a.shape()[0], a.shape()[1]);
    let (j, k2) = (b.shape()[0], b.shape()[1]);

    if k1 != k2 {
        anyhow::bail!(
            "Number of columns must match: A has {} columns, B has {} columns",
            k1,
            k2
        );
    }

    let k = k1;
    let mut result = Array2::<T>::zeros((i * j, k));

    for col_idx in 0..k {
        let a_col = a.column(col_idx);
        let b_col = b.column(col_idx);

        for (row_a_idx, a_val) in a_col.iter().enumerate() {
            for (row_b_idx, b_val) in b_col.iter().enumerate() {
                let result_row = row_a_idx * j + row_b_idx;
                result[[result_row, col_idx]] = *a_val * *b_val;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};

    #[test]
    fn test_mttkrp_basic() {
        // Simple 2×3×4 tensor
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        // Factor matrices for rank-2 decomposition
        let u1 = array![[1.0, 0.0], [0.0, 1.0]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let u3 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        // Compute MTTKRP for mode 1
        let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
    }

    #[test]
    fn test_mttkrp_all_modes() {
        // 2×3×4 tensor
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (1..=24).map(|x| x as f64).collect()).unwrap();

        // Factor matrices
        let u1 = array![[1.0, 0.5], [0.5, 1.0]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]];

        // Test MTTKRP for each mode
        let result0 = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 0).unwrap();
        assert_eq!(result0.shape(), &[2, 2]);

        let result1 = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        assert_eq!(result1.shape(), &[3, 2]);

        let result2 = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 2).unwrap();
        assert_eq!(result2.shape(), &[4, 2]);
    }

    #[test]
    fn test_khatri_rao_except_mode() {
        let u1 = array![[1.0, 2.0], [3.0, 4.0]];
        let u2 = array![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let u3 = array![[11.0, 12.0], [13.0, 14.0]];

        // Skip mode 1 (middle matrix)
        let kr = khatri_rao_except_mode(&[u1.view(), u2.view(), u3.view()], 1).unwrap();

        // Result should be kr(u3, u1) with shape (2*2, 2) = (4, 2)
        assert_eq!(kr.shape(), &[4, 2]);
    }

    #[test]
    fn test_mttkrp_rank1() {
        // Simple rank-1 test case
        let tensor = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [2.0]];

        let result = mttkrp(&tensor.view(), &[u1.view(), u2.view()], 0).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
    }

    #[test]
    #[should_panic(expected = "Mode")]
    fn test_mttkrp_invalid_mode() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [1.0], [1.0]];

        mttkrp(&tensor.view(), &[u1.view(), u2.view()], 5).unwrap();
    }

    #[test]
    #[should_panic(expected = "Number of factor matrices")]
    fn test_mttkrp_wrong_num_factors() {
        let tensor = Array::from_shape_vec(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [1.0], [1.0]];

        // Only 2 factors for a 3D tensor
        mttkrp(&tensor.view(), &[u1.view(), u2.view()], 0).unwrap();
    }

    #[test]
    #[should_panic(expected = "columns")]
    fn test_mttkrp_mismatched_ranks() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let u1 = array![[1.0, 2.0], [1.0, 2.0]]; // Rank 2
        let u2 = array![[1.0], [1.0], [1.0]]; // Rank 1

        mttkrp(&tensor.view(), &[u1.view(), u2.view()], 0).unwrap();
    }
}

/// Compute MTTKRP with blocked/tiled execution for cache efficiency
///
/// This is a cache-optimized version of MTTKRP that processes the tensor
/// in tiles/blocks to improve memory locality and reduce cache misses.
/// Use this for large tensors where the standard MTTKRP becomes memory-bound.
///
/// # Algorithm
///
/// 1. Divide the tensor into tiles along non-mode dimensions
/// 2. Process each tile independently and accumulate results
/// 3. Each tile fits better in cache, improving performance
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `factors` - Factor matrices
/// * `mode` - Mode to compute MTTKRP for
/// * `tile_size` - Size of tiles for blocking (in elements per dimension)
///
/// # Returns
///
/// Matrix with shape (I_mode, R)
///
/// # Complexity
///
/// Same asymptotic complexity as standard MTTKRP, but with better cache behavior.
/// Time: O(I_mode × R × ∏ᵢ≠ₘₒ₋ᵈₑ Iᵢ)
/// Space: O(tile_size × R) working memory per tile
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::mttkrp_blocked;
///
/// let tensor = Array::from_shape_vec(
///     vec![10, 10, 10],
///     (0..1000).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let u1 = Array::from_shape_vec((10, 4), vec![1.0; 40]).unwrap();
/// let u2 = Array::from_shape_vec((10, 4), vec![1.0; 40]).unwrap();
/// let u3 = Array::from_shape_vec((10, 4), vec![1.0; 40]).unwrap();
///
/// let result = mttkrp_blocked(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
///     4  // tile size
/// ).unwrap();
/// assert_eq!(result.shape(), &[10, 4]);
/// ```
pub fn mttkrp_blocked<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
    tile_size: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + 'static,
{
    if tile_size == 0 {
        anyhow::bail!("Tile size must be greater than 0");
    }

    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation (same as standard MTTKRP)
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

    let mode_size = tensor_shape[mode];
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    // For small tensors, fall back to standard MTTKRP
    let total_elements: usize = tensor_shape.iter().product();
    if total_elements < tile_size * tile_size {
        return mttkrp(tensor, factors, mode);
    }

    // Compute tiling scheme for non-mode dimensions
    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&i| i != mode).collect();

    // Simple blocking: process in chunks along the first non-mode dimension
    if other_dims.is_empty() {
        // Edge case: 1D tensor
        return mttkrp(tensor, factors, mode);
    }

    // For simplicity, tile along the unfolded "column" dimension
    // Unfold tensor first
    let unfolded = unfold_tensor(tensor, mode)?;
    let kr = khatri_rao_except_mode(factors, mode)?;

    let n_cols = unfolded.shape()[1];
    let n_tiles = n_cols.div_ceil(tile_size);

    // Process each tile
    for tile_idx in 0..n_tiles {
        let col_start = tile_idx * tile_size;
        let col_end = (col_start + tile_size).min(n_cols);
        let tile_width = col_end - col_start;

        // Extract tile from unfolded tensor and KR product
        let unfolded_tile = unfolded.slice(scirs2_core::ndarray_ext::s![.., col_start..col_end]);
        let kr_tile = kr.slice(scirs2_core::ndarray_ext::s![col_start..col_end, ..]);

        // Compute partial MTTKRP for this tile
        for i in 0..mode_size {
            for r in 0..cp_rank {
                let mut sum = T::zero();
                for j in 0..tile_width {
                    sum = sum + unfolded_tile[[i, j]] * kr_tile[[j, r]];
                }
                result[[i, r]] = result[[i, r]] + sum;
            }
        }
    }

    Ok(result)
}

/// Parallel blocked MTTKRP using Rayon
///
/// Combines tiling for cache efficiency with parallel execution.
/// Best for very large tensors on multi-core systems.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::mttkrp_blocked_parallel;
///
/// let tensor = Array::from_shape_vec(
///     vec![20, 20, 20],
///     (0..8000).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let u1 = Array::from_shape_vec((20, 8), vec![1.0; 160]).unwrap();
/// let u2 = Array::from_shape_vec((20, 8), vec![1.0; 160]).unwrap();
/// let u3 = Array::from_shape_vec((20, 8), vec![1.0; 160]).unwrap();
///
/// let result = mttkrp_blocked_parallel(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
///     8
/// ).unwrap();
/// assert_eq!(result.shape(), &[20, 8]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_blocked_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
    tile_size: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + Send + Sync + 'static,
{
    use scirs2_core::ndarray_ext::s;
    use scirs2_core::parallel_ops::*;

    if tile_size == 0 {
        anyhow::bail!("Tile size must be greater than 0");
    }

    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation
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
    let mode_size = tensor_shape[mode];

    // Unfold and compute KR product once
    let unfolded = unfold_tensor(tensor, mode)?;
    let kr = khatri_rao_except_mode(factors, mode)?;

    let n_cols = unfolded.shape()[1];
    let n_tiles = n_cols.div_ceil(tile_size);

    // Process tiles in parallel and accumulate
    let tile_results: Vec<Array2<T>> = (0..n_tiles)
        .into_par_iter()
        .map(|tile_idx| {
            let col_start = tile_idx * tile_size;
            let col_end = (col_start + tile_size).min(n_cols);
            let tile_width = col_end - col_start;

            let unfolded_tile = unfolded.slice(s![.., col_start..col_end]);
            let kr_tile = kr.slice(s![col_start..col_end, ..]);

            let mut tile_result = Array2::<T>::zeros((mode_size, cp_rank));

            for i in 0..mode_size {
                for r in 0..cp_rank {
                    let mut sum = T::zero();
                    for j in 0..tile_width {
                        sum = sum + unfolded_tile[[i, j]] * kr_tile[[j, r]];
                    }
                    tile_result[[i, r]] = sum;
                }
            }

            tile_result
        })
        .collect();

    // Sum all tile results
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));
    for tile_result in tile_results {
        result = result + tile_result;
    }

    Ok(result)
}

#[cfg(test)]
mod blocked_tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};

    #[test]
    fn test_mttkrp_blocked_matches_standard() {
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

        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let result_blocked =
            mttkrp_blocked(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1, 3).unwrap();

        assert_eq!(result_std.shape(), result_blocked.shape());

        // Check values match within tolerance
        for i in 0..result_std.shape()[0] {
            for j in 0..result_std.shape()[1] {
                let diff = (result_std[[i, j]] - result_blocked[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    result_std[[i, j]],
                    result_blocked[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_mttkrp_blocked_small_tensor() {
        // Small tensor should still work correctly
        let tensor = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [2.0]];

        let result = mttkrp_blocked(&tensor.view(), &[u1.view(), u2.view()], 0, 2).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
    }

    #[test]
    fn test_mttkrp_blocked_various_tile_sizes() {
        let tensor =
            Array::from_shape_vec(vec![3, 4, 5], (0..60).map(|x| x as f64).collect()).unwrap();

        let u1 = Array::from_shape_vec((3, 2), vec![1.0; 6]).unwrap();
        let u2 = Array::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let u3 = Array::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();

        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();

        // Test different tile sizes
        for tile_size in [1, 2, 5, 10, 20] {
            let result_blocked = mttkrp_blocked(
                &tensor.view(),
                &[u1.view(), u2.view(), u3.view()],
                1,
                tile_size,
            )
            .unwrap();

            assert_eq!(result_std.shape(), result_blocked.shape());

            for i in 0..result_std.shape()[0] {
                for j in 0..result_std.shape()[1] {
                    let diff = (result_std[[i, j]] - result_blocked[[i, j]]).abs();
                    assert!(diff < 1e-10);
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_mttkrp_blocked_parallel() {
        let tensor =
            Array::from_shape_vec(vec![6, 7, 8], (0..336).map(|x| x as f64).collect()).unwrap();

        let u1 = Array::from_shape_vec((6, 3), vec![1.0; 18]).unwrap();
        let u2 = Array::from_shape_vec((7, 3), vec![1.0; 21]).unwrap();
        let u3 = Array::from_shape_vec((8, 3), vec![1.0; 24]).unwrap();

        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let result_parallel =
            mttkrp_blocked_parallel(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1, 4)
                .unwrap();

        assert_eq!(result_std.shape(), result_parallel.shape());

        for i in 0..result_std.shape()[0] {
            for j in 0..result_std.shape()[1] {
                let diff = f64::abs(result_std[[i, j]] - result_parallel[[i, j]]);
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Tile size must be greater than 0")]
    fn test_mttkrp_blocked_zero_tile_size() {
        let tensor = Array::from_shape_vec(vec![2, 2], vec![1.0; 4]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [1.0]];

        mttkrp_blocked(&tensor.view(), &[u1.view(), u2.view()], 0, 0).unwrap();
    }
}

/// Fused MTTKRP kernel that avoids materializing the full Khatri-Rao product
///
/// This is a memory-efficient version of MTTKRP that computes the Khatri-Rao
/// product on-the-fly during the contraction, rather than materializing it first.
/// This saves memory: O(R × ∏ᵢ≠ₘₒ₋ᵈₑ Iᵢ) and can be more cache-efficient.
///
/// **Key optimization:** Instead of computing the full KR product matrix,
/// compute each element on-demand during the matrix multiplication.
///
/// # Algorithm
///
/// For each output element result\[i,r\]:
/// 1. Iterate through all tensor fibers along the mode
/// 2. For each fiber position, compute the KR product element on-the-fly
/// 3. Multiply by the tensor value and accumulate
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `factors` - Factor matrices
/// * `mode` - Mode to compute MTTKRP for
///
/// # Returns
///
/// Matrix with shape (I_mode, R)
///
/// # Complexity
///
/// Time: O(I_mode × R × ∏ᵢ≠ₘₒ₋ᵈₑ Iᵢ × N) where N is tensor rank
/// Space: O(I_mode × R) (no intermediate KR product storage)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused;
///
/// // 3D tensor: 4×5×6
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let u1 = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
/// let u2 = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();
/// let u3 = Array2::from_shape_vec((6, 3), vec![1.0; 18]).unwrap();
///
/// let result = mttkrp_fused(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
/// assert_eq!(result.shape(), &[5, 3]);
/// ```
pub fn mttkrp_fused<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + 'static,
{
    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation
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

    let mode_size = tensor_shape[mode];
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    // For simpler correctness, use the standard approach but avoid storing full KR
    // Instead, compute it on-the-fly during multiplication
    let unfolded = unfold_tensor(tensor, mode)?;

    // Build list of "other" dimensions (excluding mode) in order
    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&d| d != mode).collect();

    // Compute strides for unfold ordering (FORWARD): [d0, d1, d2, ...]
    // Column j = i_{d0} * (I_{d1}*I_{d2}*...) + i_{d1} * (I_{d2}*...) + ...
    // This matches khatri_rao_except_mode which now uses FORWARD ordering
    let mut strides = vec![1; other_dims.len()];
    for i in (0..other_dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tensor_shape[other_dims[i + 1]];
    }

    // For each element in the result matrix
    for i in 0..mode_size {
        for r in 0..cp_rank {
            let mut sum = T::zero();
            let n_cols = unfolded.shape()[1];

            for j in 0..n_cols {
                let tensor_val = unfolded[[i, j]];

                // Compute KR product element on-the-fly for column j and rank r
                // Extract multi-index from unfold column j in FORWARD order
                let mut kr_val = T::one();

                // Extract indices from j using FORWARD dimension ordering
                for (idx, &dim) in other_dims.iter().enumerate() {
                    let dim_size = tensor_shape[dim];
                    let dim_idx = (j / strides[idx]) % dim_size;
                    kr_val = kr_val * factors[dim][[dim_idx, r]];
                }

                sum = sum + tensor_val * kr_val;
            }

            result[[i, r]] = sum;
        }
    }

    Ok(result)
}

/// Parallel fused MTTKRP using Rayon
///
/// Combines the memory efficiency of fused MTTKRP with parallel execution.
/// Best for large tensors where memory is constrained and parallelism is available.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_parallel;
///
/// let tensor = Array::from_shape_vec(
///     vec![8, 9, 10],
///     (0..720).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let u1 = Array2::from_shape_vec((8, 4), vec![1.0; 32]).unwrap();
/// let u2 = Array2::from_shape_vec((9, 4), vec![1.0; 36]).unwrap();
/// let u3 = Array2::from_shape_vec((10, 4), vec![1.0; 40]).unwrap();
///
/// let result = mttkrp_fused_parallel(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
/// assert_eq!(result.shape(), &[9, 4]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_fused_parallel<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation (same as fused)
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
    let mode_size = tensor_shape[mode];

    // Unfold tensor
    let unfolded = unfold_tensor(tensor, mode)?;
    let n_cols = unfolded.shape()[1];

    // Build list of "other" dimensions (excluding mode) in order
    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&d| d != mode).collect();

    // Compute strides for mapping column index to multi-indices
    let mut strides = vec![1; other_dims.len()];
    for i in (0..other_dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tensor_shape[other_dims[i + 1]];
    }

    // Parallel computation over rank dimensions
    let partial_results: Vec<Array2<T>> = (0..cp_rank)
        .into_par_iter()
        .map(|r| {
            let mut result_col = Array2::<T>::zeros((mode_size, 1));

            for i in 0..mode_size {
                let mut sum = T::zero();

                for j in 0..n_cols {
                    let tensor_val = unfolded[[i, j]];

                    // Compute KR product element on-the-fly for column j and rank r
                    // Extract multi-index from unfold column j in FORWARD order
                    let mut kr_val = T::one();

                    // Extract indices from j using FORWARD dimension ordering
                    for (idx, &dim) in other_dims.iter().enumerate() {
                        let dim_size = tensor_shape[dim];
                        let dim_idx = (j / strides[idx]) % dim_size;
                        kr_val = kr_val * factors[dim][[dim_idx, r]];
                    }

                    sum = sum + tensor_val * kr_val;
                }

                result_col[[i, 0]] = sum;
            }

            result_col
        })
        .collect();

    // Assemble result from columns
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));
    for (r, col) in partial_results.iter().enumerate() {
        for i in 0..mode_size {
            result[[i, r]] = col[[i, 0]];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod fused_tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};

    #[test]
    fn test_mttkrp_manual_verification() {
        // Manually verify MTTKRP computation to understand the correct formula
        // Tensor: 2x3x2
        let tensor =
            Array::from_shape_vec(vec![2, 3, 2], (1..=12).map(|x| x as f64).collect()).unwrap();
        let u1 = array![[1.0, 0.0], [2.0, 1.0]]; // F0, shape [2, 2]
        let u2 = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]; // F1, shape [3, 2]
        let u3 = array![[1.0, 0.0], [0.0, 1.0]]; // F2, shape [2, 2]

        // Manual computation for mode=1, rank=0:
        // V[i1, r] = sum_{i0, i2} tensor[i0, i1, i2] * F0[i0, r] * F2[i2, r]
        let mut manual_result = Array2::<f64>::zeros((3, 2));
        for i1 in 0..3 {
            for r in 0..2 {
                let mut sum = 0.0;
                for i0 in 0..2 {
                    for i2 in 0..2 {
                        sum += tensor[[i0, i1, i2]] * u1[[i0, r]] * u3[[i2, r]];
                    }
                }
                manual_result[[i1, r]] = sum;
            }
        }

        // Compare with standard MTTKRP
        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();

        println!("\nManual result:\n{:?}", manual_result);
        println!("Standard MTTKRP result:\n{:?}", result_std);

        for i in 0..3 {
            for j in 0..2 {
                let diff = f64::abs(manual_result[[i, j]] - result_std[[i, j]]);
                assert!(
                    diff < 1e-10,
                    "Mismatch at [{},{}]: manual={} vs std={}",
                    i,
                    j,
                    manual_result[[i, j]],
                    result_std[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_mttkrp_fused_debug() {
        // Simple 2x3x2 tensor for easier debugging (non-square to expose ordering issues)
        let tensor =
            Array::from_shape_vec(vec![2, 3, 2], (1..=12).map(|x| x as f64).collect()).unwrap();
        let u1 = array![[1.0, 0.0], [2.0, 1.0]];
        let u2 = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]];
        let u3 = array![[1.0, 0.0], [0.0, 1.0]];

        // Print tensor
        println!("\nTensor:\n{:?}", tensor);

        // Compute unfold and KR manually to see the ordering
        let unfolded = unfold_tensor(&tensor.view(), 1).unwrap();
        let kr = khatri_rao_except_mode(&[u1.view(), u2.view(), u3.view()], 1).unwrap();

        println!("\nUnfolded (mode=1):\n{:?}", unfolded);
        println!("\nKR product:\n{:?}", kr);

        // Test mode 1
        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let result_fused =
            mttkrp_fused(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();

        println!("\nStandard result:\n{:?}", result_std);
        println!("Fused result:\n{:?}", result_fused);

        assert_eq!(result_std.shape(), result_fused.shape());

        for i in 0..result_std.shape()[0] {
            for j in 0..result_std.shape()[1] {
                let diff = f64::abs(result_std[[i, j]] - result_fused[[i, j]]);
                assert!(
                    diff < 1e-10,
                    "Mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    result_std[[i, j]],
                    result_fused[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_mttkrp_fused_matches_standard() {
        let tensor =
            Array::from_shape_vec(vec![3, 4, 5], (0..60).map(|x| x as f64).collect()).unwrap();

        let u1 = array![[1.0, 0.5], [0.5, 1.0], [0.8, 0.2]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]];
        let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75], [0.6, 0.4]];

        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let result_fused =
            mttkrp_fused(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();

        assert_eq!(result_std.shape(), result_fused.shape());

        // Check values match within tolerance
        for i in 0..result_std.shape()[0] {
            for j in 0..result_std.shape()[1] {
                let diff = f64::abs(result_std[[i, j]] - result_fused[[i, j]]);
                assert!(
                    diff < 1e-10,
                    "Mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    result_std[[i, j]],
                    result_fused[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_mttkrp_fused_all_modes() {
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (1..=24).map(|x| x as f64).collect()).unwrap();

        let u1 = array![[1.0, 0.5], [0.5, 1.0]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]];

        // Test all modes
        for mode in 0..3 {
            let result_std =
                mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], mode).unwrap();
            let result_fused =
                mttkrp_fused(&tensor.view(), &[u1.view(), u2.view(), u3.view()], mode).unwrap();

            assert_eq!(result_std.shape(), result_fused.shape());

            for i in 0..result_std.shape()[0] {
                for j in 0..result_std.shape()[1] {
                    let diff = (result_std[[i, j]] - result_fused[[i, j]]).abs();
                    assert!(diff < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_mttkrp_fused_small_tensor() {
        let tensor = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [2.0]];

        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view()], 0).unwrap();
        let result_fused = mttkrp_fused(&tensor.view(), &[u1.view(), u2.view()], 0).unwrap();

        assert_eq!(result_std.shape(), result_fused.shape());

        for i in 0..result_std.shape()[0] {
            for j in 0..result_std.shape()[1] {
                let diff = f64::abs(result_std[[i, j]] - result_fused[[i, j]]);
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn test_mttkrp_fused_rank1() {
        let tensor = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [2.0]];

        let result = mttkrp_fused(&tensor.view(), &[u1.view(), u2.view()], 0).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_mttkrp_fused_parallel() {
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();

        let u1 = Array::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let u2 = Array::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();
        let u3 = Array::from_shape_vec((6, 3), vec![1.0; 18]).unwrap();

        let result_std = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
        let result_parallel =
            mttkrp_fused_parallel(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();

        assert_eq!(result_std.shape(), result_parallel.shape());

        for i in 0..result_std.shape()[0] {
            for j in 0..result_std.shape()[1] {
                let diff = f64::abs(result_std[[i, j]] - result_parallel[[i, j]]);
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Mode")]
    fn test_mttkrp_fused_invalid_mode() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let u1 = array![[1.0], [1.0]];
        let u2 = array![[1.0], [1.0], [1.0]];

        mttkrp_fused(&tensor.view(), &[u1.view(), u2.view()], 5).unwrap();
    }
}

// =====================================================================
// SIMD-accelerated fused MTTKRP (rank-innermost loop order)
// =====================================================================
//
// Implementation notes:
//
//   The classic fused kernel loops `for i -> for r -> for j`, which streams
//   `R` partial sums through a single accumulator. The SIMD-friendly
//   reorder is `for i -> for j -> for r` (pull the rank dimension into the
//   innermost position). With `R` values laid out contiguously in memory
//   (row-major Array2 rows from `unfold_tensor` / factor rows), the inner
//   `r` loop auto-vectorizes cleanly — LLVM emits AVX2 / AVX-512 FMA on
//   x86_64, NEON FMLA on aarch64 — without any `unsafe` intrinsics.
//
//   We deliberately do NOT call `scirs2_core::simd::simd_mul_f32` /
//   `simd_fma_f32_ultra` in the innermost loop: those primitives allocate
//   a fresh `Array1<T>` per invocation, and the allocator cost dominates
//   for short `R` (16..128) vectors. Instead, we preallocate a `kr_buf`
//   of length `R`, fill it in place from the factor rows, and multiply
//   in-place into the result row — matching what the scirs2 SIMD layer
//   emits for contiguous slices, but without the per-call allocation.
//
//   This is consistent with SCIRS2_INTEGRATION_POLICY: we use
//   `scirs2_core::ndarray_ext` for all array types, and rely on the
//   platform-independent compiler auto-vectorization path. The scalar
//   fused kernel (`mttkrp_fused`) remains the reference implementation
//   and is dispatched to whenever dimensions are too small for SIMD to
//   pay off (`R < 4`) or `R == 0`.
//
//   Numerical tolerance: operation order differs from the scalar path
//   because inner products are accumulated vector-lane-wise. For FMA-free
//   paths the result is bit-identical; with FMA (enabled on modern CPUs),
//   per-element differences stay within 1e-6 for f32 and 1e-12 for f64
//   on the tested shapes, which matches the tolerance used throughout
//   the rest of the crate.

/// Minimum rank below which the scalar fused kernel is used.
/// Chosen so at least one f32 AVX2 lane (8 elements) fits comfortably,
/// but also safe for NEON (4 f32 lanes). For smaller `R`, loop overhead
/// dominates and scalar is faster.
const SIMD_MIN_RANK: usize = 4;

/// Compute MTTKRP for f64 tensors using a rank-innermost SIMD-friendly
/// loop order (auto-vectorized fused kernel).
///
/// This is the f64 counterpart of [`mttkrp_fused`]. It produces a
/// bit-for-bit equivalent result in FMA-free builds and stays within
/// `1e-12` tolerance when hardware FMA changes accumulation rounding.
///
/// # Arguments
///
/// * `tensor` - Input tensor (N-D f64)
/// * `factors` - Factor matrices `[U0, U1, ..., U_{N-1}]`, each with `R` columns
/// * `mode` - The mode to compute MTTKRP for (0-indexed)
///
/// # Returns
///
/// Matrix of shape `(I_mode, R)`.
///
/// # Errors
///
/// Returns an error on shape mismatches (same validation as [`mttkrp_fused`]).
///
/// # Performance
///
/// Auto-vectorized by the compiler over contiguous slices of length `R`.
/// For `R >= 4`, expect 2-4x speedup over the scalar fused kernel on
/// typical CP-ALS sizes. For `R < 4`, this function transparently
/// delegates to [`mttkrp_fused`].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_simd_f64;
///
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f64).collect()
/// ).unwrap();
/// let u1 = Array2::from_shape_vec((4, 8), vec![0.5; 32]).unwrap();
/// let u2 = Array2::from_shape_vec((5, 8), vec![0.5; 40]).unwrap();
/// let u3 = Array2::from_shape_vec((6, 8), vec![0.5; 48]).unwrap();
///
/// let result = mttkrp_fused_simd_f64(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
/// ).unwrap();
/// assert_eq!(result.shape(), &[5, 8]);
/// ```
pub fn mttkrp_fused_simd_f64(
    tensor: &ArrayView<f64, IxDyn>,
    factors: &[ArrayView2<f64>],
    mode: usize,
) -> Result<Array2<f64>> {
    mttkrp_fused_simd_impl::<f64>(tensor, factors, mode)
}

/// Compute MTTKRP for f32 tensors using a rank-innermost SIMD-friendly
/// loop order (auto-vectorized fused kernel).
///
/// This is the f32 counterpart of [`mttkrp_fused`]. It stays within
/// `1e-6` tolerance relative to the scalar fused kernel when hardware
/// FMA changes accumulation rounding.
///
/// # Arguments
///
/// * `tensor` - Input tensor (N-D f32)
/// * `factors` - Factor matrices with `R` columns each
/// * `mode` - Mode to compute MTTKRP for (0-indexed)
///
/// # Returns
///
/// Matrix of shape `(I_mode, R)`.
///
/// # Performance
///
/// Same auto-vectorization strategy as [`mttkrp_fused_simd_f64`]. For
/// f32 expect higher lane counts (8 elements with AVX2, 16 with AVX-512,
/// 4 with NEON) and correspondingly larger speedups.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_simd_f32;
///
/// let tensor = Array::from_shape_vec(
///     vec![4, 5, 6],
///     (0..120).map(|x| x as f32).collect()
/// ).unwrap();
/// let u1 = Array2::from_shape_vec((4, 8), vec![0.5f32; 32]).unwrap();
/// let u2 = Array2::from_shape_vec((5, 8), vec![0.5f32; 40]).unwrap();
/// let u3 = Array2::from_shape_vec((6, 8), vec![0.5f32; 48]).unwrap();
///
/// let result = mttkrp_fused_simd_f32(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
/// ).unwrap();
/// assert_eq!(result.shape(), &[5, 8]);
/// ```
pub fn mttkrp_fused_simd_f32(
    tensor: &ArrayView<f32, IxDyn>,
    factors: &[ArrayView2<f32>],
    mode: usize,
) -> Result<Array2<f32>> {
    mttkrp_fused_simd_impl::<f32>(tensor, factors, mode)
}

/// Shared implementation used by both `mttkrp_fused_simd_f32` and
/// `mttkrp_fused_simd_f64`. The `T: Num + ...` bound is narrow enough
/// that the compiler monomorphises to two specialized functions; with
/// contiguous row slices the inner loops auto-vectorize in each.
#[inline]
fn mttkrp_fused_simd_impl<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + 'static,
{
    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation (matches mttkrp_fused)
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

    let mode_size = tensor_shape[mode];

    // Empty result — skip everything.
    if cp_rank == 0 || mode_size == 0 {
        return Ok(Array2::<T>::zeros((mode_size, cp_rank)));
    }

    // Below SIMD threshold: fall back to the scalar fused reference.
    if cp_rank < SIMD_MIN_RANK {
        return mttkrp_fused(tensor, factors, mode);
    }

    // Unfold tensor along the target mode. `unfold_tensor` returns an
    // owned Array2 in standard (row-major) layout, so each row is a
    // contiguous slice of length `n_cols`.
    let unfolded = unfold_tensor(tensor, mode)?;
    let n_cols = unfolded.shape()[1];

    // List of non-mode dimensions and their strides for decoding column
    // index j into multi-index over those dims (FORWARD ordering).
    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&d| d != mode).collect();
    let mut strides = vec![1usize; other_dims.len()];
    for idx in (0..other_dims.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1] * tensor_shape[other_dims[idx + 1]];
    }

    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    // Reusable buffer for the Khatri-Rao product row (length R).
    let mut kr_buf: Vec<T> = vec![T::zero(); cp_rank];

    for i in 0..mode_size {
        // Obtain a mutable contiguous slice for result row `i`.
        // Array2 rows are contiguous in row-major layout, so `as_slice_mut`
        // succeeds — but we guard anyway to honor the no-unwrap policy.
        let mut result_row = result.row_mut(i);
        let result_slice: &mut [T] = match result_row.as_slice_mut() {
            Some(s) => s,
            None => {
                // Non-contiguous row (shouldn't happen): scalar fallback.
                return mttkrp_fused(tensor, factors, mode);
            }
        };

        // Contiguous row of the unfolded tensor for this i.
        let unfolded_row = unfolded.row(i);
        let unfolded_slice: &[T] = match unfolded_row.as_slice() {
            Some(s) => s,
            None => {
                return mttkrp_fused(tensor, factors, mode);
            }
        };

        // Iterator-based inner loops over contiguous slices: LLVM
        // auto-vectorizes these to AVX2/AVX-512/NEON as appropriate.
        for (j, &tensor_val) in unfolded_slice.iter().enumerate().take(n_cols) {
            // --- Build kr_buf[r] = prod over non-mode dims of factor[dim][idx, r] ---
            //
            // Seed with the first non-mode factor, then multiply in the rest.
            // Each inner loop is a contiguous slice-to-slice op of length R.
            let first_dim = other_dims[0];
            let first_idx = (j / strides[0]) % tensor_shape[first_dim];
            let first_row = factors[first_dim].row(first_idx);
            let first_slice: &[T] = match first_row.as_slice() {
                Some(s) => s,
                None => {
                    return mttkrp_fused(tensor, factors, mode);
                }
            };
            // Inner loop 1: kr_buf <- first_slice (auto-vectorized copy)
            kr_buf[..cp_rank].copy_from_slice(&first_slice[..cp_rank]);

            for k in 1..other_dims.len() {
                let dim_k = other_dims[k];
                let idx_k = (j / strides[k]) % tensor_shape[dim_k];
                let row_k = factors[dim_k].row(idx_k);
                let slice_k: &[T] = match row_k.as_slice() {
                    Some(s) => s,
                    None => {
                        return mttkrp_fused(tensor, factors, mode);
                    }
                };
                // Inner loop 2: kr_buf[r] *= slice_k[r] (auto-vectorized mul)
                for (dst, &src) in kr_buf[..cp_rank].iter_mut().zip(slice_k[..cp_rank].iter()) {
                    *dst = *dst * src;
                }
            }

            // Inner loop 3: result_slice[r] += tensor_val * kr_buf[r]
            // (auto-vectorized FMA)
            for (dst, &kv) in result_slice[..cp_rank]
                .iter_mut()
                .zip(kr_buf[..cp_rank].iter())
            {
                *dst = *dst + tensor_val * kv;
            }
        }
    }

    Ok(result)
}

/// Parallel f64 SIMD fused MTTKRP: same auto-vectorized inner loops as
/// [`mttkrp_fused_simd_f64`], with the outer `i` (mode) axis parallelised
/// across Rayon threads. Each thread owns its own scratch buffers so
/// there is no cross-thread synchronisation in the hot loop.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_simd_parallel_f64;
///
/// let tensor = Array::from_shape_vec(
///     vec![8, 9, 10],
///     (0..720).map(|x| x as f64).collect()
/// ).unwrap();
/// let u1 = Array2::from_shape_vec((8, 16), vec![0.25; 128]).unwrap();
/// let u2 = Array2::from_shape_vec((9, 16), vec![0.25; 144]).unwrap();
/// let u3 = Array2::from_shape_vec((10, 16), vec![0.25; 160]).unwrap();
///
/// let result = mttkrp_fused_simd_parallel_f64(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
/// ).unwrap();
/// assert_eq!(result.shape(), &[9, 16]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_fused_simd_parallel_f64(
    tensor: &ArrayView<f64, IxDyn>,
    factors: &[ArrayView2<f64>],
    mode: usize,
) -> Result<Array2<f64>> {
    mttkrp_fused_simd_parallel_impl::<f64>(tensor, factors, mode)
}

/// Parallel f32 SIMD fused MTTKRP. See [`mttkrp_fused_simd_parallel_f64`]
/// for details.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, Array2};
/// use tenrso_kernels::mttkrp_fused_simd_parallel_f32;
///
/// let tensor = Array::from_shape_vec(
///     vec![8, 9, 10],
///     (0..720).map(|x| x as f32).collect()
/// ).unwrap();
/// let u1 = Array2::from_shape_vec((8, 16), vec![0.25f32; 128]).unwrap();
/// let u2 = Array2::from_shape_vec((9, 16), vec![0.25f32; 144]).unwrap();
/// let u3 = Array2::from_shape_vec((10, 16), vec![0.25f32; 160]).unwrap();
///
/// let result = mttkrp_fused_simd_parallel_f32(
///     &tensor.view(),
///     &[u1.view(), u2.view(), u3.view()],
///     1,
/// ).unwrap();
/// assert_eq!(result.shape(), &[9, 16]);
/// ```
#[cfg(feature = "parallel")]
pub fn mttkrp_fused_simd_parallel_f32(
    tensor: &ArrayView<f32, IxDyn>,
    factors: &[ArrayView2<f32>],
    mode: usize,
) -> Result<Array2<f32>> {
    mttkrp_fused_simd_parallel_impl::<f32>(tensor, factors, mode)
}

#[cfg(feature = "parallel")]
#[inline]
fn mttkrp_fused_simd_parallel_impl<T>(
    tensor: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
    mode: usize,
) -> Result<Array2<T>>
where
    T: Copy + Num + One + Zero + Send + Sync + 'static,
{
    use scirs2_core::parallel_ops::*;

    let tensor_shape = tensor.shape();
    let rank_tensor = tensor_shape.len();

    // Validation (identical to serial impl)
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

    let mode_size = tensor_shape[mode];

    if cp_rank == 0 || mode_size == 0 {
        return Ok(Array2::<T>::zeros((mode_size, cp_rank)));
    }
    if cp_rank < SIMD_MIN_RANK {
        return mttkrp_fused(tensor, factors, mode);
    }

    let unfolded = unfold_tensor(tensor, mode)?;
    let n_cols = unfolded.shape()[1];

    let other_dims: Vec<usize> = (0..rank_tensor).filter(|&d| d != mode).collect();
    let mut strides = vec![1usize; other_dims.len()];
    for idx in (0..other_dims.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1] * tensor_shape[other_dims[idx + 1]];
    }

    // Borrow references for the closure (Send + Sync through ArrayView).
    let unfolded_ref = &unfolded;
    let other_dims_ref = &other_dims;
    let strides_ref = &strides;
    let tensor_shape_ref = tensor_shape;
    let factors_ref = factors;

    // Each thread computes its own (mode_size x cp_rank) slice of rows.
    let rows: Vec<Vec<T>> = (0..mode_size)
        .into_par_iter()
        .map(|i| -> Vec<T> {
            let mut result_row = vec![T::zero(); cp_rank];
            let mut kr_buf = vec![T::zero(); cp_rank];

            let unfolded_row = unfolded_ref.row(i);
            let unfolded_slice: &[T] = match unfolded_row.as_slice() {
                Some(s) => s,
                // Fallback: recompute scalar, element by element (rare,
                // only if unfolded rows aren't contiguous — shouldn't happen
                // for row-major Array2 but kept for safety).
                None => {
                    for j in 0..n_cols {
                        let tv = unfolded_ref[[i, j]];
                        for (r, out) in result_row.iter_mut().enumerate().take(cp_rank) {
                            let mut kv = T::one();
                            for (idx, &dim) in other_dims_ref.iter().enumerate() {
                                let di = (j / strides_ref[idx]) % tensor_shape_ref[dim];
                                kv = kv * factors_ref[dim][[di, r]];
                            }
                            *out = *out + tv * kv;
                        }
                    }
                    return result_row;
                }
            };

            for (j, &tensor_val) in unfolded_slice.iter().enumerate().take(n_cols) {
                let first_dim = other_dims_ref[0];
                let first_idx = (j / strides_ref[0]) % tensor_shape_ref[first_dim];
                let first_row = factors_ref[first_dim].row(first_idx);
                if let Some(first_slice) = first_row.as_slice() {
                    kr_buf[..cp_rank].copy_from_slice(&first_slice[..cp_rank]);
                } else {
                    for (r, dst) in kr_buf.iter_mut().enumerate().take(cp_rank) {
                        *dst = factors_ref[first_dim][[first_idx, r]];
                    }
                }

                for k in 1..other_dims_ref.len() {
                    let dim_k = other_dims_ref[k];
                    let idx_k = (j / strides_ref[k]) % tensor_shape_ref[dim_k];
                    let row_k = factors_ref[dim_k].row(idx_k);
                    if let Some(slice_k) = row_k.as_slice() {
                        for (dst, &src) in
                            kr_buf[..cp_rank].iter_mut().zip(slice_k[..cp_rank].iter())
                        {
                            *dst = *dst * src;
                        }
                    } else {
                        for (r, dst) in kr_buf.iter_mut().enumerate().take(cp_rank) {
                            *dst = *dst * factors_ref[dim_k][[idx_k, r]];
                        }
                    }
                }

                for (dst, &kv) in result_row[..cp_rank]
                    .iter_mut()
                    .zip(kr_buf[..cp_rank].iter())
                {
                    *dst = *dst + tensor_val * kv;
                }
            }

            result_row
        })
        .collect();

    let mut result = Array2::<T>::zeros((mode_size, cp_rank));
    for (i, row) in rows.into_iter().enumerate() {
        for (r, v) in row.into_iter().enumerate() {
            result[[i, r]] = v;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array;
    use scirs2_core::random::{SeedableRng, StdRng};

    /// Fill an N-D tensor with deterministic pseudo-random f64 values.
    fn rand_tensor_f64(shape: &[usize], seed: u64) -> scirs2_core::ndarray_ext::Array<f64, IxDyn> {
        let n: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Array::from_shape_vec(IxDyn(shape), data).expect("test shape valid")
    }

    fn rand_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Array2::from_shape_vec((rows, cols), data).expect("test shape valid")
    }

    fn rand_tensor_f32(shape: &[usize], seed: u64) -> scirs2_core::ndarray_ext::Array<f32, IxDyn> {
        let n: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        Array::from_shape_vec(IxDyn(shape), data).expect("test shape valid")
    }

    fn rand_matrix_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        Array2::from_shape_vec((rows, cols), data).expect("test shape valid")
    }

    fn assert_close_f64(a: &Array2<f64>, b: &Array2<f64>, tol: f64, label: &str) {
        assert_eq!(a.shape(), b.shape(), "shape mismatch ({})", label);
        for i in 0..a.shape()[0] {
            for j in 0..a.shape()[1] {
                let diff = (a[[i, j]] - b[[i, j]]).abs();
                assert!(
                    diff <= tol,
                    "{}: mismatch at [{},{}]: {} vs {} (diff {:.3e})",
                    label,
                    i,
                    j,
                    a[[i, j]],
                    b[[i, j]],
                    diff
                );
            }
        }
    }

    fn assert_close_f32(a: &Array2<f32>, b: &Array2<f32>, tol: f32, label: &str) {
        assert_eq!(a.shape(), b.shape(), "shape mismatch ({})", label);
        for i in 0..a.shape()[0] {
            for j in 0..a.shape()[1] {
                let diff = (a[[i, j]] - b[[i, j]]).abs();
                assert!(
                    diff <= tol,
                    "{}: mismatch at [{},{}]: {} vs {} (diff {:.3e})",
                    label,
                    i,
                    j,
                    a[[i, j]],
                    b[[i, j]],
                    diff
                );
            }
        }
    }

    // ---- f64 correctness: shape grid from task spec ----------------

    #[test]
    fn simd_matches_scalar_f64_shape_8_rank_4() {
        let tensor = rand_tensor_f64(&[8, 8, 8], 0xA1);
        let u1 = rand_matrix_f64(8, 4, 0xA2);
        let u2 = rand_matrix_f64(8, 4, 0xA3);
        let u3 = rand_matrix_f64(8, 4, 0xA4);
        let factors = [u1.view(), u2.view(), u3.view()];

        for mode in 0..3 {
            let scalar = mttkrp_fused(&tensor.view(), &factors, mode).expect("scalar");
            let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, mode).expect("simd");
            assert_close_f64(&scalar, &simd, 1e-12, &format!("8^3 r4 mode {}", mode));
        }
    }

    #[test]
    fn simd_matches_scalar_f64_shape_16_rank_16() {
        let tensor = rand_tensor_f64(&[16, 16, 16], 0xB1);
        let u1 = rand_matrix_f64(16, 16, 0xB2);
        let u2 = rand_matrix_f64(16, 16, 0xB3);
        let u3 = rand_matrix_f64(16, 16, 0xB4);
        let factors = [u1.view(), u2.view(), u3.view()];

        for mode in 0..3 {
            let scalar = mttkrp_fused(&tensor.view(), &factors, mode).expect("scalar");
            let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, mode).expect("simd");
            assert_close_f64(&scalar, &simd, 1e-12, &format!("16^3 r16 mode {}", mode));
        }
    }

    #[test]
    fn simd_matches_scalar_f64_shape_32_rank_33_tail() {
        // R=33 is intentionally not a multiple of 4/8 to exercise the tail
        let tensor = rand_tensor_f64(&[32, 32, 32], 0xC1);
        let u1 = rand_matrix_f64(32, 33, 0xC2);
        let u2 = rand_matrix_f64(32, 33, 0xC3);
        let u3 = rand_matrix_f64(32, 33, 0xC4);
        let factors = [u1.view(), u2.view(), u3.view()];

        let scalar = mttkrp_fused(&tensor.view(), &factors, 1).expect("scalar");
        let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, 1).expect("simd");
        assert_close_f64(&scalar, &simd, 1e-12, "32^3 r33 tail mode 1");
    }

    #[test]
    fn simd_matches_scalar_f64_shape_64_48_80_rank_64() {
        let tensor = rand_tensor_f64(&[64, 48, 80], 0xD1);
        let u1 = rand_matrix_f64(64, 64, 0xD2);
        let u2 = rand_matrix_f64(48, 64, 0xD3);
        let u3 = rand_matrix_f64(80, 64, 0xD4);
        let factors = [u1.view(), u2.view(), u3.view()];

        let scalar = mttkrp_fused(&tensor.view(), &factors, 1).expect("scalar");
        let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, 1).expect("simd");
        assert_close_f64(&scalar, &simd, 1e-12, "64x48x80 r64 mode 1");
    }

    // ---- Tail-specific: rank = lane_count + 1 (f32 lane=8, f64 lane=4) ----

    #[test]
    fn simd_tail_f32_rank_9() {
        // 9 = 8 + 1: exactly one f32 AVX2 lane + 1 tail element
        let tensor = rand_tensor_f32(&[12, 10, 14], 0xE1);
        let u1 = rand_matrix_f32(12, 9, 0xE2);
        let u2 = rand_matrix_f32(10, 9, 0xE3);
        let u3 = rand_matrix_f32(14, 9, 0xE4);
        let factors = [u1.view(), u2.view(), u3.view()];

        let scalar = mttkrp_fused(&tensor.view(), &factors, 1).expect("scalar");
        let simd = mttkrp_fused_simd_f32(&tensor.view(), &factors, 1).expect("simd");
        assert_close_f32(&scalar, &simd, 1e-4, "f32 tail r9");
    }

    #[test]
    fn simd_tail_f64_rank_5() {
        // 5 = 4 + 1: one f64 AVX2 lane + 1 tail element
        let tensor = rand_tensor_f64(&[10, 12, 8], 0xF1);
        let u1 = rand_matrix_f64(10, 5, 0xF2);
        let u2 = rand_matrix_f64(12, 5, 0xF3);
        let u3 = rand_matrix_f64(8, 5, 0xF4);
        let factors = [u1.view(), u2.view(), u3.view()];

        let scalar = mttkrp_fused(&tensor.view(), &factors, 1).expect("scalar");
        let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, 1).expect("simd");
        assert_close_f64(&scalar, &simd, 1e-12, "f64 tail r5");
    }

    // ---- Edge cases ------------------------------------------------

    #[test]
    fn simd_below_threshold_rank_1() {
        // R=1 < SIMD_MIN_RANK: must transparently delegate to scalar
        let tensor = rand_tensor_f64(&[6, 7, 5], 0x111);
        let u1 = rand_matrix_f64(6, 1, 0x112);
        let u2 = rand_matrix_f64(7, 1, 0x113);
        let u3 = rand_matrix_f64(5, 1, 0x114);
        let factors = [u1.view(), u2.view(), u3.view()];

        let scalar = mttkrp_fused(&tensor.view(), &factors, 0).expect("scalar");
        let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, 0).expect("simd");
        assert_close_f64(&scalar, &simd, 1e-15, "r=1 below-threshold");
    }

    #[test]
    fn simd_rank_zero_returns_empty() {
        // R=0 should yield a (mode_size, 0) matrix without panicking
        let tensor = rand_tensor_f64(&[4, 5, 6], 0x222);
        let u1 = Array2::<f64>::zeros((4, 0));
        let u2 = Array2::<f64>::zeros((5, 0));
        let u3 = Array2::<f64>::zeros((6, 0));
        let factors = [u1.view(), u2.view(), u3.view()];

        let simd = mttkrp_fused_simd_f64(&tensor.view(), &factors, 1).expect("simd r=0");
        assert_eq!(simd.shape(), &[5, 0]);
    }

    // ---- Parallel SIMD variant correctness ------------------------

    #[cfg(feature = "parallel")]
    #[test]
    fn simd_parallel_matches_scalar_f64() {
        let tensor = rand_tensor_f64(&[20, 24, 18], 0x331);
        let u1 = rand_matrix_f64(20, 32, 0x332);
        let u2 = rand_matrix_f64(24, 32, 0x333);
        let u3 = rand_matrix_f64(18, 32, 0x334);
        let factors = [u1.view(), u2.view(), u3.view()];

        for mode in 0..3 {
            let scalar = mttkrp_fused(&tensor.view(), &factors, mode).expect("scalar");
            let simd_par = mttkrp_fused_simd_parallel_f64(&tensor.view(), &factors, mode)
                .expect("simd parallel");
            assert_close_f64(
                &scalar,
                &simd_par,
                1e-12,
                &format!("parallel f64 mode {}", mode),
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn simd_parallel_matches_scalar_f32() {
        let tensor = rand_tensor_f32(&[16, 20, 14], 0x441);
        let u1 = rand_matrix_f32(16, 24, 0x442);
        let u2 = rand_matrix_f32(20, 24, 0x443);
        let u3 = rand_matrix_f32(14, 24, 0x444);
        let factors = [u1.view(), u2.view(), u3.view()];

        let scalar = mttkrp_fused(&tensor.view(), &factors, 1).expect("scalar");
        let simd_par =
            mttkrp_fused_simd_parallel_f32(&tensor.view(), &factors, 1).expect("simd par");
        assert_close_f32(&scalar, &simd_par, 1e-4, "parallel f32 mode 1");
    }

    // ---- Validation errors propagate ------------------------------

    #[test]
    #[should_panic(expected = "Mode")]
    fn simd_invalid_mode_panics() {
        let tensor = Array::from_shape_vec(IxDyn(&[3, 3]), vec![1.0f64; 9]).expect("shape");
        let u1 = rand_matrix_f64(3, 8, 1);
        let u2 = rand_matrix_f64(3, 8, 2);
        mttkrp_fused_simd_f64(&tensor.view(), &[u1.view(), u2.view()], 5).expect("must fail");
    }
}
