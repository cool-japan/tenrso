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
    T: Clone + Num + One + Zero,
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
    let mode_size = tensor_shape[mode];
    let mut result = Array2::<T>::zeros((mode_size, cp_rank));

    for i in 0..mode_size {
        for r in 0..cp_rank {
            let mut sum = T::zero();
            for j in 0..kr.shape()[0] {
                sum = sum + unfolded[[i, j]].clone() * kr[[j, r]].clone();
            }
            result[[i, r]] = sum;
        }
    }

    Ok(result)
}

/// Unfold a tensor along a specific mode into a matrix
fn unfold_tensor<T>(tensor: &ArrayView<T, IxDyn>, mode: usize) -> Result<Array2<T>>
where
    T: Clone + Num,
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
/// The matrices are multiplied in reverse order (last to first, skipping mode)
fn khatri_rao_except_mode<T>(factors: &[ArrayView2<T>], skip_mode: usize) -> Result<Array2<T>>
where
    T: Clone + Num,
{
    // Collect all matrices except the one at skip_mode, in reverse order
    let mut matrices: Vec<&ArrayView2<T>> = Vec::new();
    for i in (0..factors.len()).rev() {
        if i != skip_mode {
            matrices.push(&factors[i]);
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
    T: Clone + Num,
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
                result[[result_row, col_idx]] = a_val.clone() * b_val.clone();
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
    T: Clone + Num + One + Zero,
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
                    sum = sum + unfolded_tile[[i, j]].clone() * kr_tile[[j, r]].clone();
                }
                result[[i, r]] = result[[i, r]].clone() + sum;
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
    T: Clone + Num + One + Zero + Send + Sync,
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
                        sum = sum + unfolded_tile[[i, j]].clone() * kr_tile[[j, r]].clone();
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
                let diff = (result_std[[i, j]] - result_parallel[[i, j]]).abs();
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
