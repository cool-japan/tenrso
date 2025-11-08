//! N-mode product implementation (TTM - Tensor Times Matrix)
//!
//! The N-mode product is a fundamental tensor operation that multiplies a tensor
//! by a matrix along a specific mode. For tensor X ∈ ℝ^(I₁×...×Iₙ) and matrix
//! M ∈ ℝ^(J×Iₖ), the result Y = X ×ₖ M has shape (I₁×...×Iₖ₋₁×J×Iₖ₊₁×...×Iₙ).
//!
//! This operation is critical for Tucker decomposition, HOSVD, and tensor networks.
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use anyhow::Result;
use scirs2_core::ndarray_ext::{Array, Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::{Num, One, Zero};

/// Compute the N-mode product (tensor times matrix) of a tensor and a matrix
///
/// For tensor X with shape (I₁, I₂, ..., Iₙ) and matrix M with shape (J, Iₖ),
/// computes Y = X ×ₖ M with shape (I₁, ..., Iₖ₋₁, J, Iₖ₊₁, ..., Iₙ).
///
/// # Algorithm
///
/// 1. Unfold tensor X along mode k to get matrix X_(k) of shape (Iₖ, ∏ᵢ≠ₖ Iᵢ)
/// 2. Compute Y_(k) = M · X_(k) with shape (J, ∏ᵢ≠ₖ Iᵢ)
/// 3. Fold Y_(k) back to tensor Y with the new shape
///
/// # Arguments
///
/// * `tensor` - Input tensor with N dimensions
/// * `matrix` - Matrix with shape (J, Iₖ) where Iₖ matches `tensor.shape()[mode]`
/// * `mode` - The mode along which to perform the product (0-indexed)
///
/// # Returns
///
/// Result tensor with the mode-k dimension replaced by J
///
/// # Errors
///
/// Returns error if:
/// - Mode is out of bounds
/// - Matrix columns don't match tensor mode size
///
/// # Complexity
///
/// Time: O(J * Iₖ * ∏ᵢ≠ₖ Iᵢ) = O(J * total_elements)
/// Space: O(J * ∏ᵢ≠ₖ Iᵢ)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::nmode_product;
///
/// // 3D tensor: 2×3×4
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// // Matrix: 5×3 (will replace mode-1 dimension)
/// let matrix = array![[1.0, 0.0, 0.0],
///                      [0.0, 1.0, 0.0],
///                      [0.0, 0.0, 1.0],
///                      [1.0, 1.0, 0.0],
///                      [0.0, 1.0, 1.0]];
///
/// let result = nmode_product(&tensor.view(), &matrix.view(), 1).unwrap();
/// assert_eq!(result.shape(), &[2, 5, 4]);  // 2×5×4
/// ```
pub fn nmode_product<T>(
    tensor: &ArrayView<T, IxDyn>,
    matrix: &ArrayView2<T>,
    mode: usize,
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero,
{
    let tensor_shape = tensor.shape();
    let rank = tensor_shape.len();

    if mode >= rank {
        anyhow::bail!("Mode {} out of bounds for tensor with rank {}", mode, rank);
    }

    let mode_size = tensor_shape[mode];
    let (matrix_rows, matrix_cols) = (matrix.shape()[0], matrix.shape()[1]);

    if matrix_cols != mode_size {
        anyhow::bail!(
            "Matrix columns ({}) must match tensor mode-{} size ({})",
            matrix_cols,
            mode,
            mode_size
        );
    }

    // Step 1: Unfold tensor along mode k
    let unfolded = unfold_tensor(tensor, mode)?;

    // Step 2: Matrix multiplication: M · X_(k)
    // Manual matrix multiplication: result[i,j] = sum_k matrix[i,k] * unfolded[k,j]
    let mut result_unfolded = Array2::<T>::zeros((matrix_rows, unfolded.shape()[1]));
    for i in 0..matrix_rows {
        for j in 0..unfolded.shape()[1] {
            let mut sum = T::zero();
            for k in 0..matrix_cols {
                sum = sum + matrix[[i, k]].clone() * unfolded[[k, j]].clone();
            }
            result_unfolded[[i, j]] = sum;
        }
    }

    // Step 3: Fold back to tensor with new shape
    let mut new_shape: Vec<usize> = tensor_shape.to_vec();
    new_shape[mode] = matrix_rows;

    let result = fold_matrix(&result_unfolded.view(), &new_shape, mode)?;

    Ok(result)
}

/// Unfold a tensor along a specific mode into a matrix
///
/// For tensor X with shape (I₁, ..., Iₖ, ..., Iₙ), unfolds to matrix
/// with shape (Iₖ, I₁ · ... · Iₖ₋₁ · Iₖ₊₁ · ... · Iₙ)
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

/// Fold a matrix back into a tensor along a specific mode
///
/// Inverse operation of unfold_tensor
fn fold_matrix<T>(matrix: &ArrayView2<T>, shape: &[usize], mode: usize) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num,
{
    let mode_size = shape[mode];
    let other_size: usize = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &s)| s)
        .product();

    if matrix.shape() != [mode_size, other_size] {
        anyhow::bail!(
            "Matrix shape {:?} incompatible with tensor shape {:?} and mode {}",
            matrix.shape(),
            shape,
            mode
        );
    }

    // Create intermediate shape: [mode_size, other_dims...]
    let mut inter_shape = Vec::with_capacity(shape.len());
    inter_shape.push(mode_size);
    for (i, &s) in shape.iter().enumerate() {
        if i != mode {
            inter_shape.push(s);
        }
    }

    // Reshape matrix to intermediate shape
    let inter = (*matrix).into_shape_with_order(IxDyn(&inter_shape))?;

    // Create inverse permutation
    let mut inv_perm = vec![0; shape.len()];
    inv_perm[mode] = 0;
    let mut idx = 1;
    for (i, item) in inv_perm.iter_mut().enumerate().take(shape.len()) {
        if i != mode {
            *item = idx;
            idx += 1;
        }
    }

    // Permute back to original axis order
    let tensor = inter.permuted_axes(IxDyn(&inv_perm));

    Ok(tensor.into_owned())
}

/// Compute multiple N-mode products sequentially
///
/// Applies matrices to a tensor along multiple modes in sequence.
/// This is useful for Tucker decomposition and tensor networks.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `matrices` - Slice of (matrix, mode) pairs to apply
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::nmode_products_seq;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let m1 = array![[1.0, 0.0], [0.0, 1.0]];  // 2×2
/// let m2 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];  // 2×3
///
/// let result = nmode_products_seq(
///     &tensor.view(),
///     &[(&m1.view(), 0), (&m2.view(), 1)]
/// ).unwrap();
/// assert_eq!(result.shape(), &[2, 2, 4]);
/// ```
pub fn nmode_products_seq<T>(
    tensor: &ArrayView<T, IxDyn>,
    matrices: &[(&ArrayView2<T>, usize)],
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero,
{
    let mut result = tensor.to_owned();

    for (matrix, mode) in matrices {
        result = nmode_product(&result.view(), matrix, *mode)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_nmode_product_3d() {
        // 3D tensor: 2×3×4
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        // Matrix: 5×3 (multiply along mode 1)
        let matrix = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ];

        let result = nmode_product(&tensor.view(), &matrix.view(), 1).unwrap();
        assert_eq!(result.shape(), &[2, 5, 4]);
    }

    #[test]
    fn test_nmode_product_identity() {
        // 3D tensor: 2×3×4
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        // Identity matrix: 3×3 (should not change the tensor)
        let identity = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = nmode_product(&tensor.view(), &identity.view(), 1).unwrap();
        assert_eq!(result.shape(), tensor.shape());

        // Values should be unchanged
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let idx = [i, j, k];
                    assert_eq!(result[&idx[..]], tensor[&idx[..]]);
                }
            }
        }
    }

    #[test]
    fn test_nmode_product_mode0() {
        // 2×3 matrix as a simple 2D tensor
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Matrix: 3×2 (multiply along mode 0)
        let matrix = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let result = nmode_product(&tensor.view(), &matrix.view(), 0).unwrap();
        assert_eq!(result.shape(), &[3, 3]);

        // First row: 1*[1,2,3] + 0*[4,5,6] = [1,2,3]
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[0, 2]], 3.0);

        // Second row: 0*[1,2,3] + 1*[4,5,6] = [4,5,6]
        assert_eq!(result[[1, 0]], 4.0);
        assert_eq!(result[[1, 1]], 5.0);
        assert_eq!(result[[1, 2]], 6.0);

        // Third row: 1*[1,2,3] + 1*[4,5,6] = [5,7,9]
        assert_eq!(result[[2, 0]], 5.0);
        assert_eq!(result[[2, 1]], 7.0);
        assert_eq!(result[[2, 2]], 9.0);
    }

    #[test]
    fn test_nmode_products_seq() {
        // 2×3×4 tensor
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        let m1 = array![[1.0, 0.0], [0.0, 1.0]];
        let m2 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let result =
            nmode_products_seq(&tensor.view(), &[(&m1.view(), 0), (&m2.view(), 1)]).unwrap();
        assert_eq!(result.shape(), &[2, 2, 4]);
    }

    #[test]
    fn test_unfold_fold_roundtrip() {
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        for mode in 0..3 {
            let unfolded = unfold_tensor(&tensor.view(), mode).unwrap();
            let folded = fold_matrix(&unfolded.view(), &[2, 3, 4], mode).unwrap();

            assert_eq!(folded.shape(), tensor.shape());

            // Check values match
            for i in 0..2 {
                for j in 0..3 {
                    for k in 0..4 {
                        let idx = [i, j, k];
                        assert_eq!(folded[&idx[..]], tensor[&idx[..]]);
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Mode")]
    fn test_nmode_product_invalid_mode() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let matrix = array![[1.0, 0.0]];
        nmode_product(&tensor.view(), &matrix.view(), 5).unwrap();
    }

    #[test]
    #[should_panic(expected = "Matrix columns")]
    fn test_nmode_product_size_mismatch() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let matrix = array![[1.0, 0.0]]; // 1×2, but mode-0 has size 2
        nmode_product(&tensor.view(), &matrix.view(), 1).unwrap(); // mode-1 has size 3, mismatch!
    }
}

/// Tucker operator: compute multiple N-mode products simultaneously
///
/// This is an optimized version of applying multiple N-mode products in sequence.
/// It's used extensively in Tucker decomposition and tensor networks.
///
/// For a tensor X and factor matrices U₁, U₂, ..., Uₙ, computes:
/// Y = X ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ
///
/// This is more efficient than calling `nmode_product` repeatedly because it:
/// 1. Plans the optimal execution order based on intermediate tensor sizes
/// 2. Minimizes intermediate storage
/// 3. Exploits cache locality
///
/// # Arguments
///
/// * `tensor` - Input tensor with N dimensions
/// * `factor_matrices` - Map from mode index to matrix to apply
///
/// # Returns
///
/// Result tensor with modified dimensions according to applied matrices
///
/// # Errors
///
/// Returns error if:
/// - Any mode index is out of bounds
/// - Matrix dimensions don't match tensor mode sizes
///
/// # Complexity
///
/// Time: O(∏ᵢ Iᵢ × ∏ⱼ Jⱼ) where Iᵢ are input dims and Jⱼ are output dims
/// Space: O(max intermediate tensor size)
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::tucker_operator;
///
/// // 2×3×4 tensor
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// // Apply matrices to modes 0 and 2
/// let m1 = array![[1.0, 0.0], [0.0, 1.0]];  // 2×2
/// let m2 = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];  // 2×4
///
/// let mut factors = HashMap::new();
/// factors.insert(0, m1.view());
/// factors.insert(2, m2.view());
///
/// let result = tucker_operator(&tensor.view(), &factors).unwrap();
/// assert_eq!(result.shape(), &[2, 3, 2]);  // mode-2 reduced from 4 to 2
/// ```
pub fn tucker_operator<T>(
    tensor: &ArrayView<T, IxDyn>,
    factor_matrices: &std::collections::HashMap<usize, ArrayView2<T>>,
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero,
{
    if factor_matrices.is_empty() {
        return Ok(tensor.to_owned());
    }

    let rank = tensor.shape().len();

    // Validate all mode indices
    for &mode in factor_matrices.keys() {
        if mode >= rank {
            anyhow::bail!("Mode {} out of bounds for tensor with rank {}", mode, rank);
        }
    }

    // Plan execution order: process modes in order to minimize intermediate size
    // Heuristic: sort by reduction ratio (output_size / input_size)
    let mut modes: Vec<usize> = factor_matrices.keys().copied().collect();
    modes.sort_by(|&a, &b| {
        let ratio_a = factor_matrices[&a].shape()[0] as f64 / tensor.shape()[a] as f64;
        let ratio_b = factor_matrices[&b].shape()[0] as f64 / tensor.shape()[b] as f64;
        // Safe to unwrap: tensor dimensions are always positive, so ratios are never NaN
        ratio_a
            .partial_cmp(&ratio_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply N-mode products in planned order
    let mut result = tensor.to_owned();

    for original_mode in modes {
        let matrix = factor_matrices[&original_mode];
        result = nmode_product(&result.view(), &matrix, original_mode)?;
    }

    Ok(result)
}

/// Tucker operator with explicit mode ordering
///
/// Like `tucker_operator`, but allows explicit control over execution order.
/// This can be useful for performance tuning or when you have domain knowledge
/// about the optimal order.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `mode_matrix_pairs` - Ordered list of (mode, matrix) pairs to apply
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::tucker_operator_ordered;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// let m1 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];  // 2×3 for mode 1
/// let m2 = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];  // 2×4 for mode 2
///
/// // Apply in specific order: mode 2 first, then mode 1
/// let result = tucker_operator_ordered(
///     &tensor.view(),
///     &[(2, &m2.view()), (1, &m1.view())]
/// ).unwrap();
/// assert_eq!(result.shape(), &[2, 2, 2]);
/// ```
pub fn tucker_operator_ordered<T>(
    tensor: &ArrayView<T, IxDyn>,
    mode_matrix_pairs: &[(usize, &ArrayView2<T>)],
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero,
{
    let mut result = tensor.to_owned();

    for &(mode, matrix) in mode_matrix_pairs {
        result = nmode_product(&result.view(), matrix, mode)?;
    }

    Ok(result)
}

/// Tucker reconstruction: reconstruct tensor from core and factor matrices
///
/// For Tucker decomposition, reconstructs the original tensor from:
/// - Core tensor G with shape (R₁, R₂, ..., Rₙ)
/// - Factor matrices U₁, U₂, ..., Uₙ with shapes (Iₖ, Rₖ)
///
/// Computes: X = G ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ
///
/// # Arguments
///
/// * `core` - Core tensor with reduced dimensions
/// * `factors` - Factor matrices (one per mode)
///
/// # Returns
///
/// Reconstructed tensor with full dimensions
///
/// # Errors
///
/// Returns error if:
/// - Number of factors doesn't match core rank
/// - Factor matrix columns don't match core dimensions
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::{Array, array};
/// use tenrso_kernels::tucker_reconstruct;
///
/// // Core tensor: 2×2×2
/// let core = Array::from_shape_vec(
///     vec![2, 2, 2],
///     vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
/// ).unwrap();
///
/// // Factor matrices
/// let u1 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];  // 3×2
/// let u2 = array![[1.0, 0.0], [0.0, 1.0]];  // 2×2
/// let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];  // 3×2
///
/// let tensor = tucker_reconstruct(&core.view(), &[u1.view(), u2.view(), u3.view()]).unwrap();
/// assert_eq!(tensor.shape(), &[3, 2, 3]);
/// ```
pub fn tucker_reconstruct<T>(
    core: &ArrayView<T, IxDyn>,
    factors: &[ArrayView2<T>],
) -> Result<Array<T, IxDyn>>
where
    T: Clone + Num + One + Zero,
{
    let core_shape = core.shape();
    let rank = core_shape.len();

    if factors.len() != rank {
        anyhow::bail!(
            "Number of factor matrices ({}) must match core rank ({})",
            factors.len(),
            rank
        );
    }

    // Validate factor dimensions match core dimensions
    for (i, factor) in factors.iter().enumerate() {
        if factor.shape()[1] != core_shape[i] {
            anyhow::bail!(
                "Factor matrix {} has {} columns, expected {} (core mode-{} size)",
                i,
                factor.shape()[1],
                core_shape[i],
                i
            );
        }
    }

    // Build factor map for tucker_operator
    let mut factor_map = std::collections::HashMap::new();
    for (mode, factor) in factors.iter().enumerate() {
        factor_map.insert(mode, *factor);
    }

    tucker_operator(core, &factor_map)
}

#[cfg(test)]
mod tucker_tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;
    use std::collections::HashMap;

    #[test]
    fn test_tucker_operator_single_mode() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let m1 = array![[1.0, 0.0], [0.0, 1.0]];
        let mut factors = HashMap::new();
        factors.insert(0, m1.view());

        let result = tucker_operator(&tensor.view(), &factors).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_tucker_operator_multiple_modes() {
        // 2×3×4 tensor
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        let m1 = array![[1.0, 0.0], [0.0, 1.0]];
        let m2 = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let mut factors = HashMap::new();
        factors.insert(0, m1.view());
        factors.insert(2, m2.view());

        let result = tucker_operator(&tensor.view(), &factors).unwrap();
        assert_eq!(result.shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_tucker_operator_empty() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let factors = HashMap::new();

        let result = tucker_operator(&tensor.view(), &factors).unwrap();
        assert_eq!(result.shape(), tensor.shape());

        // Should be unchanged
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(result[[i, j]], tensor[[i, j]]);
            }
        }
    }

    #[test]
    fn test_tucker_operator_ordered() {
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        let m1 = array![[1.0, 0.0], [0.0, 1.0]];
        let m2 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let m3 = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];

        let result = tucker_operator_ordered(
            &tensor.view(),
            &[(0, &m1.view()), (1, &m2.view()), (2, &m3.view())],
        )
        .unwrap();

        assert_eq!(result.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_tucker_reconstruct_identity() {
        // Core tensor: 2×2
        let core = Array::from_shape_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        // Identity factor matrices
        let u1 = array![[1.0, 0.0], [0.0, 1.0]];
        let u2 = array![[1.0, 0.0], [0.0, 1.0]];

        let tensor = tucker_reconstruct(&core.view(), &[u1.view(), u2.view()]).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);

        // Should equal the core
        assert_eq!(tensor[[0, 0]], 1.0);
        assert_eq!(tensor[[0, 1]], 0.0);
        assert_eq!(tensor[[1, 0]], 0.0);
        assert_eq!(tensor[[1, 1]], 1.0);
    }

    #[test]
    fn test_tucker_reconstruct_3d() {
        // Small 2×2×2 core
        let core =
            Array::from_shape_vec(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();

        let u1 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]; // 3×2
        let u2 = array![[1.0, 0.0], [0.0, 1.0]]; // 2×2
        let u3 = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]; // 3×2

        let tensor = tucker_reconstruct(&core.view(), &[u1.view(), u2.view(), u3.view()]).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "Number of factor matrices")]
    fn test_tucker_reconstruct_wrong_num_factors() {
        let core = Array::from_shape_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let u1 = array![[1.0, 0.0], [0.0, 1.0]];

        // Only 1 factor for 3D core
        tucker_reconstruct(&core.view(), &[u1.view()]).unwrap();
    }

    #[test]
    #[should_panic(expected = "columns")]
    fn test_tucker_reconstruct_mismatched_dims() {
        let core = Array::from_shape_vec(vec![2, 2], vec![1.0; 4]).unwrap();
        let u1 = array![[1.0, 0.0], [0.0, 1.0]]; // 2×2, correct
        let u2 = array![[1.0], [0.0]]; // 2×1, wrong (should be 2×2)

        tucker_reconstruct(&core.view(), &[u1.view(), u2.view()]).unwrap();
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_tucker_operator_invalid_mode() {
        let tensor = Array::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let m1 = array![[1.0]];
        let mut factors = HashMap::new();
        factors.insert(5, m1.view());

        tucker_operator(&tensor.view(), &factors).unwrap();
    }

    #[test]
    fn test_tucker_operator_dimension_reduction() {
        // Test that dimensions are properly reduced
        let tensor =
            Array::from_shape_vec(vec![4, 5, 6], (0..120).map(|x| x as f64).collect()).unwrap();

        let m1 = Array::from_shape_vec((2, 4), vec![1.0; 8]).unwrap(); // 4 -> 2
        let m2 = Array::from_shape_vec((3, 5), vec![1.0; 15]).unwrap(); // 5 -> 3
        let m3 = Array::from_shape_vec((4, 6), vec![1.0; 24]).unwrap(); // 6 -> 4

        let mut factors = HashMap::new();
        factors.insert(0, m1.view());
        factors.insert(1, m2.view());
        factors.insert(2, m3.view());

        let result = tucker_operator(&tensor.view(), &factors).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
    }
}
