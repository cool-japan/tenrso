//! Tensor contraction operations
//!
//! This module provides general tensor contraction primitives that are fundamental
//! for tensor network computations, quantum simulations, and advanced tensor operations.
//!
//! # Operations
//!
//! - **Pairwise contraction** - Contract two tensors along specified modes
//! - **Mode contraction** - Sum a tensor along specific modes (trace-like operations)
//! - **Inner product** - Compute inner product between two tensors
//! - **Trace operations** - Compute traces along mode pairs
//!
//! # SciRS2 Integration
//!
//! All array operations use `scirs2_core::ndarray_ext`.
//! Direct use of `ndarray` is forbidden per SCIRS2_INTEGRATION_POLICY.md

use crate::error::{KernelError, KernelResult};
use scirs2_core::ndarray_ext::{Array, ArrayView, Axis, IxDyn, Zip};
use scirs2_core::numeric::{Num, Zero};

/// Contract two tensors along specified mode pairs
///
/// Performs a generalized contraction of two tensors by summing over shared modes.
/// For tensors A (shape I₁×I₂×...×Iₙ) and B (shape J₁×J₂×...×Jₘ), contracting
/// modes (i, j) means summing over the indices where A[..., i, ...] = B[..., j, ...].
///
/// # Arguments
///
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `modes_a` - Modes of tensor A to contract
/// * `modes_b` - Modes of tensor B to contract (must match modes_a in length and dimensions)
///
/// # Returns
///
/// Contracted tensor with remaining modes from both inputs
///
/// # Errors
///
/// Returns error if:
/// - Mode indices out of bounds
/// - Number of contraction modes mismatch
/// - Dimension mismatch along contracted modes
///
/// # Complexity
///
/// Time: O(∏(free_dims) × ∏(contracted_dims))
/// Space: O(∏(free_dims))
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::contract_tensors;
///
/// // Matrix multiplication as tensor contraction
/// let a = Array::from_shape_vec(vec![2, 3], (0..6).map(|x| x as f64).collect()).unwrap();
/// let b = Array::from_shape_vec(vec![3, 4], (0..12).map(|x| x as f64).collect()).unwrap();
///
/// // Contract mode 1 of A with mode 0 of B (matrix multiplication)
/// let result = contract_tensors(&a.view(), &b.view(), &[1], &[0]).unwrap();
/// assert_eq!(result.shape(), &[2, 4]);
/// ```
pub fn contract_tensors<T>(
    a: &ArrayView<T, IxDyn>,
    b: &ArrayView<T, IxDyn>,
    modes_a: &[usize],
    modes_b: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Num + Zero,
{
    // Validate inputs
    if modes_a.len() != modes_b.len() {
        return Err(KernelError::operation_error(
            "contract_tensors",
            format!(
                "Number of contraction modes must match: {} vs {}",
                modes_a.len(),
                modes_b.len()
            ),
        ));
    }

    let shape_a = a.shape();
    let shape_b = b.shape();

    // Validate mode indices and dimensions
    for (&mode_a, &mode_b) in modes_a.iter().zip(modes_b.iter()) {
        if mode_a >= shape_a.len() {
            return Err(KernelError::invalid_mode(
                mode_a,
                shape_a.len(),
                "contract_tensors: mode_a out of bounds",
            ));
        }
        if mode_b >= shape_b.len() {
            return Err(KernelError::invalid_mode(
                mode_b,
                shape_b.len(),
                "contract_tensors: mode_b out of bounds",
            ));
        }
        if shape_a[mode_a] != shape_b[mode_b] {
            return Err(KernelError::incompatible_shapes(
                "contract_tensors",
                shape_a.to_vec(),
                shape_b.to_vec(),
                format!(
                    "Contracted dimensions must match: {} vs {}",
                    shape_a[mode_a], shape_b[mode_b]
                ),
            ));
        }
    }

    // Determine free modes and output shape
    let free_modes_a: Vec<usize> = (0..shape_a.len())
        .filter(|m| !modes_a.contains(m))
        .collect();
    let free_modes_b: Vec<usize> = (0..shape_b.len())
        .filter(|m| !modes_b.contains(m))
        .collect();

    let mut output_shape = Vec::new();
    for &mode in &free_modes_a {
        output_shape.push(shape_a[mode]);
    }
    for &mode in &free_modes_b {
        output_shape.push(shape_b[mode]);
    }

    // Handle scalar output
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    let mut result = Array::zeros(IxDyn(&output_shape));

    // Compute contraction via explicit summation
    // For better performance, this could be optimized with unfold+GEMM for specific cases
    contract_explicit(
        a,
        b,
        modes_a,
        modes_b,
        &free_modes_a,
        &free_modes_b,
        &mut result,
    )?;

    Ok(result)
}

/// Helper function to perform explicit contraction
fn contract_explicit<T>(
    a: &ArrayView<T, IxDyn>,
    b: &ArrayView<T, IxDyn>,
    modes_a: &[usize],
    modes_b: &[usize],
    free_modes_a: &[usize],
    free_modes_b: &[usize],
    result: &mut Array<T, IxDyn>,
) -> KernelResult<()>
where
    T: Clone + Num + Zero,
{
    let shape_a = a.shape();
    let shape_b = b.shape();

    // Get contracted dimensions
    let contracted_dims: Vec<usize> = modes_a.iter().map(|&m| shape_a[m]).collect();

    // Compute total contracted size
    let contracted_size: usize = contracted_dims.iter().product();

    // Create index iterators for output dimensions
    let free_dims_a: Vec<usize> = free_modes_a.iter().map(|&m| shape_a[m]).collect();
    let _free_dims_b: Vec<usize> = free_modes_b.iter().map(|&m| shape_b[m]).collect();

    // Iterate over all output indices
    for out_idx in 0..result.len() {
        let mut sum = T::zero();

        // Convert linear index to multi-dimensional for output
        let out_coord = linear_to_coord(out_idx, result.shape());

        // Split output coordinates into A and B parts
        let n_free_a = free_dims_a.len();
        let coord_a_free = &out_coord[..n_free_a];
        let coord_b_free = &out_coord[n_free_a..];

        // Sum over contracted indices
        for contract_idx in 0..contracted_size {
            let contract_coord = linear_to_coord(contract_idx, &contracted_dims);

            // Build full coordinates for A and B
            let mut coord_a = vec![0; shape_a.len()];
            let mut coord_b = vec![0; shape_b.len()];

            for (i, &mode) in free_modes_a.iter().enumerate() {
                coord_a[mode] = coord_a_free[i];
            }
            for (i, &mode) in modes_a.iter().enumerate() {
                coord_a[mode] = contract_coord[i];
            }

            for (i, &mode) in free_modes_b.iter().enumerate() {
                coord_b[mode] = coord_b_free[i];
            }
            for (i, &mode) in modes_b.iter().enumerate() {
                coord_b[mode] = contract_coord[i];
            }

            let val_a = a[&coord_a[..]].clone();
            let val_b = b[&coord_b[..]].clone();
            sum = sum + val_a * val_b;
        }

        result[&out_coord[..]] = sum;
    }

    Ok(())
}

/// Convert linear index to multi-dimensional coordinates
fn linear_to_coord(mut linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coord = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        coord[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
    coord
}

/// Sum a tensor along specified modes (trace-like operation)
///
/// Reduces a tensor by summing over the specified modes, similar to a generalized trace.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `modes` - Modes to sum over (0-indexed)
///
/// # Returns
///
/// Tensor with summed modes removed
///
/// # Errors
///
/// Returns error if mode indices are out of bounds
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::sum_over_modes;
///
/// let tensor = Array::from_shape_vec(
///     vec![2, 3, 4],
///     (0..24).map(|x| x as f64).collect()
/// ).unwrap();
///
/// // Sum over mode 1 (middle dimension)
/// let result = sum_over_modes(&tensor.view(), &[1]).unwrap();
/// assert_eq!(result.shape(), &[2, 4]);
/// ```
pub fn sum_over_modes<T>(
    tensor: &ArrayView<T, IxDyn>,
    modes: &[usize],
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Num + Zero,
{
    let shape = tensor.shape();

    // Validate modes
    for &mode in modes {
        if mode >= shape.len() {
            return Err(KernelError::invalid_mode(
                mode,
                shape.len(),
                "sum_over_modes",
            ));
        }
    }

    // Determine output shape (remove summed modes)
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|m| !modes.contains(m))
        .map(|m| shape[m])
        .collect();

    if output_shape.is_empty() {
        // Sum to scalar
        let total: T = tensor.iter().cloned().fold(T::zero(), |acc, x| acc + x);
        return Ok(Array::from_shape_vec(vec![1], vec![total]).unwrap());
    }

    // Sum along specified axes
    let mut result = tensor.to_owned();
    let mut sorted_modes: Vec<usize> = modes.to_vec();
    sorted_modes.sort_unstable_by(|a, b| b.cmp(a)); // Sort in descending order

    for &mode in &sorted_modes {
        result = result.sum_axis(Axis(mode));
    }

    Ok(result)
}

/// Compute inner product (Frobenius inner product) between two tensors
///
/// Computes ⟨A, B⟩ = ∑ᵢ Aᵢ × Bᵢ (element-wise product then sum)
///
/// # Arguments
///
/// * `a` - First input tensor
/// * `b` - Second input tensor
///
/// # Returns
///
/// Scalar value representing the inner product
///
/// # Errors
///
/// Returns error if tensor shapes don't match
///
/// # Complexity
///
/// Time: O(tensor_size)
/// Space: O(1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::tensor_inner_product;
///
/// let a = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = Array::from_shape_vec(vec![2, 3], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
///
/// let inner = tensor_inner_product(&a.view(), &b.view()).unwrap();
/// assert_eq!(inner, 21.0); // 1+2+3+4+5+6 = 21
/// ```
pub fn tensor_inner_product<T>(a: &ArrayView<T, IxDyn>, b: &ArrayView<T, IxDyn>) -> KernelResult<T>
where
    T: Clone + Num + Zero,
{
    if a.shape() != b.shape() {
        return Err(KernelError::incompatible_shapes(
            "tensor_inner_product",
            a.shape().to_vec(),
            b.shape().to_vec(),
            "Tensors must have the same shape",
        ));
    }

    let mut sum = T::zero();
    Zip::from(a).and(b).for_each(|a_val, b_val| {
        sum = sum.clone() + a_val.clone() * b_val.clone();
    });

    Ok(sum)
}

/// Compute trace of a tensor along two modes
///
/// For a tensor with equal dimensions along two modes, computes the sum
/// over diagonal elements where those two mode indices are equal.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `mode1` - First mode for trace
/// * `mode2` - Second mode for trace
///
/// # Returns
///
/// Tensor with the traced modes removed
///
/// # Errors
///
/// Returns error if:
/// - Modes are out of bounds
/// - Modes are the same
/// - Dimensions along modes don't match
///
/// # Complexity
///
/// Time: O(tensor_size / dim)
/// Space: O(output_size)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::Array;
/// use tenrso_kernels::tensor_trace;
///
/// // 3×3×2 tensor
/// let tensor = Array::from_shape_vec(
///     vec![3, 3, 2],
///     (0..18).map(|x| x as f64).collect()
/// ).unwrap();
///
/// // Trace over modes 0 and 1 (sum diagonal: [0,0,:], [1,1,:], [2,2,:])
/// let result = tensor_trace(&tensor.view(), 0, 1).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// ```
pub fn tensor_trace<T>(
    tensor: &ArrayView<T, IxDyn>,
    mode1: usize,
    mode2: usize,
) -> KernelResult<Array<T, IxDyn>>
where
    T: Clone + Num + Zero,
{
    let shape = tensor.shape();

    if mode1 >= shape.len() {
        return Err(KernelError::invalid_mode(
            mode1,
            shape.len(),
            "tensor_trace: mode1",
        ));
    }
    if mode2 >= shape.len() {
        return Err(KernelError::invalid_mode(
            mode2,
            shape.len(),
            "tensor_trace: mode2",
        ));
    }
    if mode1 == mode2 {
        return Err(KernelError::operation_error(
            "tensor_trace",
            "Trace modes must be different",
        ));
    }
    if shape[mode1] != shape[mode2] {
        return Err(KernelError::operation_error(
            "tensor_trace",
            format!(
                "Trace modes must have equal dimensions: {} vs {}",
                shape[mode1], shape[mode2]
            ),
        ));
    }

    let trace_dim = shape[mode1];

    // Determine output shape (remove both traced modes)
    let output_shape: Vec<usize> = (0..shape.len())
        .filter(|&m| m != mode1 && m != mode2)
        .map(|m| shape[m])
        .collect();

    if output_shape.is_empty() {
        // Trace to scalar
        let mut total = T::zero();
        for i in 0..trace_dim {
            let mut coord = vec![0; shape.len()];
            coord[mode1] = i;
            coord[mode2] = i;
            total = total.clone() + tensor[&coord[..]].clone();
        }
        return Ok(Array::from_shape_vec(vec![1], vec![total]).unwrap());
    }

    let mut result = Array::zeros(IxDyn(&output_shape));

    // Iterate over all output coordinates
    for out_idx in 0..result.len() {
        let out_coord = linear_to_coord(out_idx, &output_shape);

        let mut sum = T::zero();
        for diag_idx in 0..trace_dim {
            // Build full coordinate
            let mut coord = Vec::new();
            let mut out_pos = 0;

            for m in 0..shape.len() {
                if m == mode1 || m == mode2 {
                    coord.push(diag_idx);
                } else {
                    coord.push(out_coord[out_pos]);
                    out_pos += 1;
                }
            }

            sum = sum.clone() + tensor[&coord[..]].clone();
        }

        result[&out_coord[..]] = sum;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::{array, Array};

    #[test]
    fn test_contract_tensors_matmul() {
        // Matrix multiplication: A (2×3) × B (3×4) = C (2×4)
        let a = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array::from_shape_vec(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();

        let result = contract_tensors(&a.view(), &b.view(), &[1], &[0]).unwrap();
        assert_eq!(result.shape(), &[2, 4]);

        // First row should be [1, 2, 3, 0]
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[0, 2]], 3.0);
    }

    #[test]
    fn test_contract_tensors_3d() {
        // Contract 3D tensors
        let a = Array::<f64, _>::ones(IxDyn(&[2, 3, 4]));
        let b = Array::<f64, _>::ones(IxDyn(&[4, 5]));

        // Contract last mode of A with first mode of B
        let result = contract_tensors(&a.view(), &b.view(), &[2], &[0]).unwrap();
        assert_eq!(result.shape(), &[2, 3, 5]);

        // Each element should be 4.0 (sum of 4 ones)
        assert_eq!(result[[0, 0, 0]], 4.0);
    }

    #[test]
    fn test_sum_over_modes_single() {
        let tensor =
            Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f64).collect()).unwrap();

        let result = sum_over_modes(&tensor.view(), &[1]).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
    }

    #[test]
    fn test_sum_over_modes_multiple() {
        let tensor = Array::<f64, _>::ones(IxDyn(&[2, 3, 4]));

        let result = sum_over_modes(&tensor.view(), &[0, 2]).unwrap();
        assert_eq!(result.shape(), &[3]);

        // Each element should be 2*4 = 8
        for i in 0..3 {
            assert_eq!(result[[i]], 8.0);
        }
    }

    #[test]
    fn test_sum_over_modes_to_scalar() {
        let tensor = Array::<f64, _>::ones(IxDyn(&[2, 3, 4]));

        let result = sum_over_modes(&tensor.view(), &[0, 1, 2]).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result[[0]], 24.0);
    }

    #[test]
    fn test_tensor_inner_product() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 0.0], [0.0, 1.0]];

        let a_dyn = a.into_dyn();
        let b_dyn = b.into_dyn();

        let inner = tensor_inner_product(&a_dyn.view(), &b_dyn.view()).unwrap();
        assert_eq!(inner, 5.0); // 1*1 + 2*0 + 3*0 + 4*1 = 5
    }

    #[test]
    fn test_tensor_inner_product_ones() {
        let a = Array::<f64, _>::ones(IxDyn(&[3, 4, 5]));
        let b = Array::<f64, _>::ones(IxDyn(&[3, 4, 5]));

        let inner = tensor_inner_product(&a.view(), &b.view()).unwrap();
        assert_eq!(inner, 60.0); // 3*4*5 = 60
    }

    #[test]
    fn test_tensor_trace_matrix() {
        // 3×3 identity matrix
        let tensor = Array::from_shape_vec(
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        let result = tensor_trace(&tensor.view(), 0, 1).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result[[0]], 3.0); // trace of identity is dimension
    }

    #[test]
    fn test_tensor_trace_3d() {
        // 2×2×3 tensor
        let tensor =
            Array::from_shape_vec(vec![2, 2, 3], (0..12).map(|x| x as f64).collect()).unwrap();

        // Trace over first two modes
        let result = tensor_trace(&tensor.view(), 0, 1).unwrap();
        assert_eq!(result.shape(), &[3]);

        // Diagonal elements: [0,0,:] and [1,1,:]
        // [0,0,0]=0, [0,0,1]=1, [0,0,2]=2
        // [1,1,0]=9, [1,1,1]=10, [1,1,2]=11
        // Trace: [9, 11, 13]
        assert_eq!(result[[0]], 9.0);
        assert_eq!(result[[1]], 11.0);
        assert_eq!(result[[2]], 13.0);
    }

    #[test]
    fn test_contract_error_dimension_mismatch() {
        let a = Array::<f64, _>::ones(IxDyn(&[2, 3]));
        let b = Array::<f64, _>::ones(IxDyn(&[4, 5]));

        let result = contract_tensors(&a.view(), &b.view(), &[1], &[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_error_same_mode() {
        let tensor = Array::<f64, _>::ones(IxDyn(&[3, 3]));

        let result = tensor_trace(&tensor.view(), 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_error_dimension_mismatch() {
        let tensor = Array::<f64, _>::ones(IxDyn(&[2, 3]));

        let result = tensor_trace(&tensor.view(), 0, 1);
        assert!(result.is_err());
    }
}
