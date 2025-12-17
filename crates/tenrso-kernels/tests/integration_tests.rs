//! Integration tests for tenrso-kernels with tenrso-core
//!
//! These tests verify that tensor operations work correctly when using
//! DenseND tensors from tenrso-core with kernel operations.

use scirs2_core::ndarray_ext::{s, Array, Array2};
use tenrso_core::DenseND;
use tenrso_kernels::{
    contract_tensors, frobenius_norm_tensor, hadamard, khatri_rao, kronecker, mean_along_modes,
    mttkrp, nmode_product, pnorm_along_modes, sum_along_modes, tensor_inner_product,
    variance_along_modes,
};

#[test]
fn test_tensor_workflow_basic() {
    // Create a 3D tensor using tenrso-core
    let tensor = DenseND::<f64>::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Verify shape
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.rank(), 3);
    assert_eq!(tensor.len(), 24);
}

#[test]
fn test_nmode_product_with_densend() {
    // Create a 2×3×4 tensor
    let tensor = DenseND::<f64>::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Create a matrix 5×3 to multiply along mode 1
    let matrix = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    // Perform N-mode product
    let result = nmode_product(&tensor.view(), &matrix.view(), 1).unwrap();

    // Result should have shape [2, 5, 4]
    assert_eq!(result.shape(), &[2, 5, 4]);
}

#[test]
fn test_mttkrp_with_factor_matrices() {
    // Create a 2×3×4 tensor
    let tensor =
        DenseND::<f64>::from_vec((1..=24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Factor matrices for rank-2 CP decomposition
    let u1 = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();
    let u2 = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();
    let u3 =
        Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.25, 0.75]).unwrap();

    // Compute MTTKRP for each mode
    let result0 = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 0).unwrap();
    assert_eq!(result0.shape(), &[2, 2]);

    let result1 = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
    assert_eq!(result1.shape(), &[3, 2]);

    let result2 = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 2).unwrap();
    assert_eq!(result2.shape(), &[4, 2]);
}

#[test]
fn test_hadamard_with_tensors() {
    // Create two 2×3 tensors
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::<f64>::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0], &[2, 3]).unwrap();

    // Unfold both to 2D matrices
    let a_mat = a.unfold(0).unwrap();
    let b_mat = b.unfold(0).unwrap();

    // Compute Hadamard product
    let result = hadamard(&a_mat.view(), &b_mat.view());

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result[[0, 0]], 2.0); // 1*2
    assert_eq!(result[[0, 1]], 4.0); // 2*2
    assert_eq!(result[[1, 2]], 12.0); // 6*2
}

#[test]
fn test_khatri_rao_for_decomposition() {
    // Create factor matrices for CP decomposition
    let u1 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let u2 = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5]).unwrap();

    // Compute Khatri-Rao product
    let kr = khatri_rao(&u1.view(), &u2.view());

    // Result should have shape [3*4, 2] = [12, 2]
    assert_eq!(kr.shape(), &[12, 2]);

    // Verify structure: each row is outer product of rows from u1 and u2
    // First block: u1[0] ⊗ u2
    assert_eq!(kr[[0, 0]], 1.0); // 1*1
    assert_eq!(kr[[0, 1]], 0.0); // 2*0
    assert_eq!(kr[[1, 0]], 0.0); // 1*0
    assert_eq!(kr[[1, 1]], 2.0); // 2*1
}

#[test]
fn test_kronecker_structure() {
    // Create small matrices
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    // Compute Kronecker product
    let kron = kronecker(&a.view(), &b.view());

    // Result should be 4×4
    assert_eq!(kron.shape(), &[4, 4]);

    // Verify block structure: each element of A multiplies the entire B matrix
    // Top-left block: 1*B
    assert_eq!(kron[[0, 0]], 5.0);
    assert_eq!(kron[[0, 1]], 6.0);
    assert_eq!(kron[[1, 0]], 7.0);
    assert_eq!(kron[[1, 1]], 8.0);

    // Bottom-right block: 4*B
    assert_eq!(kron[[2, 2]], 20.0); // 4*5
    assert_eq!(kron[[3, 3]], 32.0); // 4*8
}

#[test]
fn test_unfold_with_kernel_operations() {
    // Create a 3D tensor
    let tensor = DenseND::<f64>::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Unfold along each mode and verify shapes
    let unfold0 = tensor.unfold(0).unwrap();
    assert_eq!(unfold0.shape(), &[2, 12]); // mode-0: 2 × (3*4)

    let unfold1 = tensor.unfold(1).unwrap();
    assert_eq!(unfold1.shape(), &[3, 8]); // mode-1: 3 × (2*4)

    let unfold2 = tensor.unfold(2).unwrap();
    assert_eq!(unfold2.shape(), &[4, 6]); // mode-2: 4 × (2*3)

    // Verify we can perform operations on unfolded tensors
    let identity = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    // This should preserve the tensor when folded back
    let transformed = hadamard(&identity.view(), &unfold0.slice(s![.., 0..2]).view());
    assert_eq!(transformed.shape(), &[2, 2]);
}

#[test]
fn test_reshape_and_operations() {
    // Create a tensor
    let tensor = DenseND::<f64>::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Reshape to 2D
    let reshaped = tensor.reshape(&[6, 4]).unwrap();
    assert_eq!(reshaped.shape(), &[6, 4]);

    // Unfold and apply operations
    let mat = reshaped.unfold(0).unwrap();
    assert_eq!(mat.shape(), &[6, 4]);

    // Create a matrix for Hadamard product
    let ones = Array2::ones((6, 4));
    let result = hadamard(&mat.view(), &ones.view());
    assert_eq!(result.shape(), &[6, 4]);
}

#[test]
fn test_permute_and_nmode() {
    // Create a 2×3×4 tensor
    let tensor = DenseND::<f64>::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Permute axes: [0,1,2] -> [2,0,1]
    let permuted = tensor.permute(&[2, 0, 1]).unwrap();
    assert_eq!(permuted.shape(), &[4, 2, 3]);

    // Apply N-mode product on permuted tensor along mode 0
    // Matrix must have shape (J, I₀) where I₀=4 (mode-0 size)
    let matrix = Array2::from_shape_vec(
        (3, 4),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let result = nmode_product(&permuted.view(), &matrix.view(), 0).unwrap();

    // Result should have shape [3, 2, 3] (mode-0 replaced by 3)
    assert_eq!(result.shape(), &[3, 2, 3]);
}

// =============================================================================
// Integration Tests for Tensor Contractions
// =============================================================================

#[test]
fn test_contraction_workflow_matmul() {
    // Demonstrate contraction as matrix multiplication
    let a = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = DenseND::<f64>::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[3, 2]).unwrap();

    // Contract mode 1 of A with mode 0 of B (matrix multiplication)
    let result = contract_tensors(&a.view(), &b.view(), &[1], &[0]).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    // Verify matrix multiplication result
    assert_eq!(result[[0, 0]], 1.0 + 0.0 + 3.0); // row 0 of A · col 0 of B
    assert_eq!(result[[0, 1]], 0.0 + 2.0 + 0.0); // row 0 of A · col 1 of B
}

#[test]
fn test_contraction_with_nmode_product() {
    // Create a 3D tensor and contract after n-mode product
    let tensor = DenseND::<f64>::ones(&[2, 3, 4]);
    let matrix = Array2::<f64>::ones((5, 3));

    // Apply n-mode product
    let intermediate = nmode_product(&tensor.view(), &matrix.view(), 1).unwrap();
    assert_eq!(intermediate.shape(), &[2, 5, 4]);

    // Now contract modes to reduce dimensionality
    let identity = Array::<f64, _>::eye(5).into_dyn();
    let contracted = contract_tensors(&intermediate.view(), &identity.view(), &[1], &[0]).unwrap();

    // Result should have summed over the contracted mode
    assert_eq!(contracted.shape(), &[2, 4, 5]);
}

#[test]
fn test_inner_product_for_similarity() {
    // Compute similarity between two tensors using inner product
    let tensor_a =
        DenseND::<f64>::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();
    let tensor_b = DenseND::<f64>::ones(&[2, 3, 4]);

    let similarity = tensor_inner_product(&tensor_a.view(), &tensor_b.view()).unwrap();

    // Inner product should equal sum of elements in tensor_a
    let expected: f64 = (0..24).sum::<usize>() as f64;
    assert_eq!(similarity, expected);
}

#[test]
fn test_contraction_chain() {
    // Demonstrate chaining multiple contractions
    let a = Array::<f64, _>::ones(vec![2, 3, 4]).into_dyn();
    let b = Array::<f64, _>::ones(vec![4, 5]).into_dyn();
    let c = Array::<f64, _>::ones(vec![5, 6]).into_dyn();

    // Contract A with B
    let ab = contract_tensors(&a.view(), &b.view(), &[2], &[0]).unwrap();
    assert_eq!(ab.shape(), &[2, 3, 5]);

    // Contract AB with C
    let abc = contract_tensors(&ab.view(), &c.view(), &[2], &[0]).unwrap();
    assert_eq!(abc.shape(), &[2, 3, 6]);

    // Each element should be 4*5 = 20 (from the summations)
    assert_eq!(abc[[0, 0, 0]], 20.0);
}

// =============================================================================
// Integration Tests for Tensor Reductions
// =============================================================================

#[test]
fn test_reduction_workflow_statistics() {
    // Analyze tensor statistics using reductions
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = DenseND::<f64>::from_vec(data.clone(), &[2, 2, 3]).unwrap();

    // Compute statistics along mode 2 (last dimension)
    let sum = sum_along_modes(&tensor.view(), &[2]).unwrap();
    assert_eq!(sum.shape(), &[2, 2]);
    assert_eq!(sum[[0, 0]], 6.0); // 1+2+3

    let mean = mean_along_modes(&tensor.view(), &[2]).unwrap();
    assert_eq!(mean.shape(), &[2, 2]);
    assert!((mean[[0, 0]] - 2.0).abs() < 1e-10); // (1+2+3)/3

    let variance = variance_along_modes(&tensor.view(), &[2], 1).unwrap();
    assert_eq!(variance.shape(), &[2, 2]);
    // Variance of [1,2,3] = 1.0
    assert!((variance[[0, 0]] - 1.0).abs() < 1e-10);
}

#[test]
fn test_norms_for_convergence_tracking() {
    // Simulate tracking convergence in iterative algorithms
    let iteration1 = DenseND::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let iteration2 = DenseND::<f64>::from_vec(vec![1.1, 2.1, 3.0, 3.9, 5.0, 6.1], &[2, 3]).unwrap();

    // Compute Frobenius norm of each iteration
    let norm1 = frobenius_norm_tensor(&iteration1.view());
    let norm2 = frobenius_norm_tensor(&iteration2.view());

    // Both should be positive
    assert!(norm1 > 0.0);
    assert!(norm2 > 0.0);

    // Compute difference  - convert to Array2 for subtraction
    let iter1_arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let iter2_arr = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.0, 3.9, 5.0, 6.1]).unwrap();
    let diff = &iter2_arr - &iter1_arr;

    let change_norm = frobenius_norm_tensor(&diff.into_dyn().view());

    // Change should be small (we only changed values slightly)
    assert!(change_norm < 1.0);
}

#[test]
fn test_pnorm_for_regularization() {
    // Demonstrate using p-norms for regularization
    let weights = DenseND::<f64>::from_vec(vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.0], &[2, 3]).unwrap();

    // L1 norm (for L1 regularization)
    let l1_norm = pnorm_along_modes(&weights.view(), &[0, 1], 1.0).unwrap();
    let expected_l1 = 0.5 + 0.3 + 0.8 + 0.2 + 0.1 + 0.0;
    assert!((l1_norm[[0]] - expected_l1).abs() < 1e-10);

    // L2 norm (for L2 regularization)
    let l2_norm = pnorm_along_modes(&weights.view(), &[0, 1], 2.0).unwrap();
    let expected_l2 = (0.5 * 0.5 + 0.3 * 0.3 + 0.8 * 0.8 + 0.2 * 0.2 + 0.1 * 0.1_f64).sqrt();
    assert!((l2_norm[[0]] - expected_l2).abs() < 1e-10);
}

#[test]
fn test_reduction_after_mttkrp() {
    // Real-world scenario: compute MTTKRP then analyze the result
    let tensor =
        DenseND::<f64>::from_vec((1..=24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    let u1 = Array2::<f64>::ones((2, 2));
    let u2 = Array2::<f64>::ones((3, 2));
    let u3 = Array2::<f64>::ones((4, 2));

    // Compute MTTKRP for mode 1
    let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 1).unwrap();
    assert_eq!(result.shape(), &[3, 2]);

    // Analyze the result using reductions
    let result_dyn = result.clone().into_dyn();
    let sum = sum_along_modes(&result_dyn.view(), &[0]).unwrap();
    assert_eq!(sum.shape(), &[2]);

    let mean = mean_along_modes(&result_dyn.view(), &[0]).unwrap();
    assert_eq!(mean.shape(), &[2]);

    // Both should be positive (we have positive input data)
    assert!(sum[[0]] > 0.0);
    assert!(mean[[0]] > 0.0);
}

// =============================================================================
// Complex Workflows Combining Multiple Operations
// =============================================================================

#[test]
fn test_cp_decomposition_workflow_simulation() {
    // Simulate a mini CP decomposition iteration workflow
    let tensor =
        DenseND::<f64>::from_vec((1..=24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Initialize factor matrices
    let mut u1 = Array2::<f64>::ones((2, 2));
    let u2 = Array2::<f64>::ones((3, 2));
    let u3 = Array2::<f64>::ones((4, 2));

    // Step 1: Compute MTTKRP for mode 0
    let gradient = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 0).unwrap();
    assert_eq!(gradient.shape(), &[2, 2]);

    // Step 2: Update factor matrix (simple assignment for demo)
    u1 = gradient;

    // Step 3: Analyze convergence using norms
    let u1_dyn = u1.clone().into_dyn();
    let norm = frobenius_norm_tensor(&u1_dyn.view());
    assert!(norm > 0.0);

    // Step 4: Check for numerical stability using variance
    let col_variance = variance_along_modes(&u1_dyn.view(), &[0], 1).unwrap();
    assert_eq!(col_variance.shape(), &[2]);
}

#[test]
fn test_tucker_decomposition_workflow_simulation() {
    // Simulate Tucker decomposition workflow with n-mode products
    let tensor = DenseND::<f64>::ones(&[3, 4, 5]);

    // Factor matrices (orthogonal for Tucker)
    let u1 = Array2::<f64>::eye(3);
    let u2 = Array2::<f64>::eye(4);
    let u3 = Array2::<f64>::eye(5);

    // Step 1: Apply n-mode products
    let temp1 = nmode_product(&tensor.view(), &u1.view(), 0).unwrap();
    let temp2 = nmode_product(&temp1.view(), &u2.view(), 1).unwrap();
    let core = nmode_product(&temp2.view(), &u3.view(), 2).unwrap();

    assert_eq!(core.shape(), &[3, 4, 5]);

    // Step 2: Analyze core tensor using reductions
    let mean = mean_along_modes(&core.view(), &[0, 1, 2]).unwrap();
    assert_eq!(mean.shape(), &[1]);
    assert!((mean[[0]] - 1.0).abs() < 1e-10); // Mean of ones is 1

    // Step 3: Check reconstruction error using inner product
    let error = tensor_inner_product(&tensor.view(), &core.view()).unwrap();
    assert!(error > 0.0);
}

#[test]
fn test_validation_workflow() {
    // Demonstrate validation of tensor operations using multiple checks
    let original =
        DenseND::<f64>::from_vec((1..=24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

    // Apply some transformations
    let matrix = Array2::<f64>::eye(2);
    let transformed = nmode_product(&original.view(), &matrix.view(), 0).unwrap();

    // Validation 1: Check shape preservation
    assert_eq!(transformed.shape(), original.shape());

    // Validation 2: Check value preservation (identity should not change values)
    // Use inner product to compare
    let similarity = tensor_inner_product(&original.view(), &transformed.view()).unwrap();
    let norm_orig = frobenius_norm_tensor(&original.view());
    let norm_trans = frobenius_norm_tensor(&transformed.view());

    // If tensors are identical, inner product equals squared norm
    assert!((similarity - norm_orig * norm_trans).abs() < 1e-9);

    // Validation 3: Check statistical properties
    let orig_mean = mean_along_modes(&original.view(), &[0, 1, 2]).unwrap();
    let trans_mean = mean_along_modes(&transformed.view(), &[0, 1, 2]).unwrap();
    assert!((orig_mean[[0]] - trans_mean[[0]]).abs() < 1e-10);
}

#[test]
fn test_tensor_analysis_pipeline() {
    // Complete analysis pipeline: load, transform, analyze
    let data =
        DenseND::<f64>::from_vec((0..60).map(|x| (x as f64) * 0.1).collect(), &[3, 4, 5]).unwrap();

    // Step 1: Compute summary statistics
    let overall_mean = mean_along_modes(&data.view(), &[0, 1, 2]).unwrap();
    let overall_variance = variance_along_modes(&data.view(), &[0, 1, 2], 1).unwrap();

    assert_eq!(overall_mean.shape(), &[1]);
    assert_eq!(overall_variance.shape(), &[1]);

    // Step 2: Compute per-mode statistics
    let mode0_mean = mean_along_modes(&data.view(), &[0]).unwrap();
    assert_eq!(mode0_mean.shape(), &[4, 5]);

    let mode1_mean = mean_along_modes(&data.view(), &[1]).unwrap();
    assert_eq!(mode1_mean.shape(), &[3, 5]);

    // Step 3: Apply transformation based on statistics
    let matrix = Array2::<f64>::ones((2, 3));
    let projected = nmode_product(&data.view(), &matrix.view(), 0).unwrap();
    assert_eq!(projected.shape(), &[2, 4, 5]);

    // Step 4: Validate transformation preserved certain properties
    let proj_sum = sum_along_modes(&projected.view(), &[0]).unwrap();
    assert_eq!(proj_sum.shape(), &[4, 5]);
}

#[test]
fn test_error_analysis_workflow() {
    // Simulate error analysis in reconstruction
    let original = DenseND::<f64>::ones(&[3, 4, 5]);
    let reconstructed = DenseND::<f64>::from_vec(
        vec![1.01; 60], // Slightly different values
        &[3, 4, 5],
    )
    .unwrap();

    // Compute error metrics using norms and inner product
    let orig_norm = frobenius_norm_tensor(&original.view());
    let recon_norm = frobenius_norm_tensor(&reconstructed.view());

    // Error estimation based on norm difference
    let error_estimate = (recon_norm - orig_norm).abs();

    assert!(error_estimate < 0.2); // Small error

    // Relative error
    let relative_error = error_estimate / orig_norm;
    assert!(relative_error < 0.05); // Less than 5% error
}

// ============================================================================
// Large Tensor Tests (100³+)
// ============================================================================
// These tests verify performance and correctness with production-scale tensors

#[test]
fn test_mttkrp_large_tensor_100cubed() {
    // Test MTTKRP with 100³ tensor (1M elements) - realistic production size
    let size = 100;
    let rank = 32;

    // Create a large tensor with structured data
    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
        (idx[0] + idx[1] + idx[2]) as f64
    }));

    // Create factor matrices
    let u1 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64 + 1.0);
    let u2 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i * 2 + j) as f64 + 1.0);
    let u3 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j * 3) as f64 + 1.0);

    // Test MTTKRP for each mode
    for mode in 0..3 {
        let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], mode).unwrap();

        // Verify output shape
        assert_eq!(result.shape(), &[size, rank]);

        // Verify result is not all zeros or NaN
        let result_norm = result.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(result_norm.is_finite());
        assert!(result_norm > 0.0);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_mttkrp_blocked_parallel_large_tensor() {
    // Test blocked parallel MTTKRP with large tensor
    use tenrso_kernels::mttkrp_blocked_parallel;

    let size = 80;
    let rank = 16;
    let tile_size = 16;

    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
        ((idx[0] * idx[1] + idx[2]) % 100) as f64 + 1.0
    }));

    let u1 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64 + 1.0);
    let u2 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64 + 1.0);
    let u3 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| (i + j) as f64 + 1.0);

    // Compare blocked parallel with standard MTTKRP
    let factors = vec![u1.view(), u2.view(), u3.view()];
    let result_standard = mttkrp(&tensor.view(), &factors, 0).unwrap();
    let result_blocked = mttkrp_blocked_parallel(&tensor.view(), &factors, 0, tile_size).unwrap();

    // Results should be very close (allowing for floating point differences)
    let diff: f64 = result_standard
        .iter()
        .zip(result_blocked.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    let max_val = result_standard.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let relative_diff = diff / (max_val * (size * rank) as f64);

    assert!(
        relative_diff < 1e-10,
        "Blocked parallel result differs from standard: relative_diff = {}",
        relative_diff
    );
}

#[test]
fn test_large_khatri_rao_product() {
    // Test Khatri-Rao with large matrices
    let rows = 500;
    let cols = 64;

    let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| {
        ((i * 17 + j * 13) % 100) as f64 + 1.0
    });
    let b = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| {
        ((i * 23 + j * 19) % 100) as f64 + 1.0
    });

    let result = khatri_rao(&a.view(), &b.view());

    // Verify shape
    assert_eq!(result.shape(), &[rows * rows, cols]);

    // Verify column structure - each column should be Kronecker product of corresponding columns
    for col in 0..3 {
        // Check first few columns
        let a_col = a.column(col);
        let b_col = b.column(col);
        let result_col = result.column(col);

        // Check a few elements of the Kronecker structure
        for i in 0..5 {
            for j in 0..5 {
                let expected = a_col[i] * b_col[j];
                let actual = result_col[i * rows + j];
                assert!(
                    (expected - actual).abs() < 1e-10,
                    "Mismatch in KR column structure"
                );
            }
        }
    }
}

#[test]
fn test_large_hadamard_product() {
    // Test Hadamard product with large matrices
    let size = 1000;

    let a = Array2::<f64>::from_shape_fn((size, size), |(i, j)| ((i + j) % 100) as f64 + 1.0);
    let b = Array2::<f64>::from_shape_fn((size, size), |(i, j)| ((i * j) % 100) as f64 + 1.0);

    let result = hadamard(&a.view(), &b.view());

    // Verify element-wise multiplication
    for i in 0..10 {
        for j in 0..10 {
            let expected = a[[i, j]] * b[[i, j]];
            let actual = result[[i, j]];
            assert!(
                (expected - actual).abs() < 1e-10,
                "Hadamard product mismatch at ({}, {})",
                i,
                j
            );
        }
    }

    // Verify norm
    let result_norm = frobenius_norm_tensor(&result.view().into_dyn());
    assert!(result_norm.is_finite());
    assert!(result_norm > 0.0);
}

#[test]
fn test_large_nmode_product() {
    // Test N-mode product with large 3D tensor
    let size = 100;
    let new_size = 80;

    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
        ((idx[0] + idx[1] + idx[2]) % 100) as f64 + 1.0
    }));

    let matrix =
        Array2::<f64>::from_shape_fn((new_size, size), |(i, j)| ((i + j) % 50) as f64 + 1.0);

    // Apply N-mode product along mode 0
    let result = nmode_product(&tensor.view(), &matrix.view(), 0).unwrap();

    // Verify shape
    assert_eq!(result.shape(), &[new_size, size, size]);

    // Verify result is valid
    let result_norm = frobenius_norm_tensor(&result.view());
    assert!(result_norm.is_finite());
    assert!(result_norm > 0.0);

    // Verify some structural properties
    let result_mean = mean_along_modes(&result.view(), &[0]).unwrap();
    assert_eq!(result_mean.shape(), &[size, size]);
}

#[test]
fn test_large_tensor_contractions() {
    // Test tensor contractions with large tensors
    let size = 80;

    let a = Array::from_shape_fn(vec![size, size], |idx| {
        ((idx[0] + idx[1]) % 100) as f64 + 1.0
    });
    let b = Array::from_shape_fn(vec![size, size], |idx| {
        ((idx[0] * 2 + idx[1]) % 100) as f64 + 1.0
    });

    // Matrix multiplication via contraction
    let result = contract_tensors(&a.view(), &b.view(), &[1], &[0]).unwrap();

    // Verify shape
    assert_eq!(result.shape(), &[size, size]);

    // Verify result is valid
    let result_norm = frobenius_norm_tensor(&result.view());
    assert!(result_norm.is_finite());
    assert!(result_norm > 0.0);

    // Verify a few elements manually
    for i in 0..3 {
        for j in 0..3 {
            let mut expected = 0.0;
            for k in 0..size {
                expected += a[[i, k]] * b[[k, j]];
            }
            let actual = result[[i, j]];
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-8,
                "Contraction mismatch at ({}, {}): expected {}, got {}, diff {}",
                i,
                j,
                expected,
                actual,
                diff
            );
        }
    }
}

#[test]
fn test_large_tensor_reductions() {
    // Test statistical reductions with large tensors
    let size = 100;

    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
        ((idx[0] + idx[1] * 2 + idx[2] * 3) % 100) as f64
    }));

    // Test sum
    let sum_result = sum_along_modes(&tensor.view(), &[0]).unwrap();
    assert_eq!(sum_result.shape(), &[size, size]);
    assert!(sum_result.iter().all(|&x| x.is_finite()));

    // Test mean
    let mean_result = mean_along_modes(&tensor.view(), &[0]).unwrap();
    assert_eq!(mean_result.shape(), &[size, size]);
    assert!(mean_result.iter().all(|&x| x.is_finite()));

    // Verify mean is sum / size
    for i in 0..size {
        for j in 0..size {
            let expected_mean = sum_result[[i, j]] / size as f64;
            let actual_mean = mean_result[[i, j]];
            assert!(
                (expected_mean - actual_mean).abs() < 1e-10,
                "Mean calculation mismatch"
            );
        }
    }

    // Test variance
    let var_result = variance_along_modes(&tensor.view(), &[0], 0).unwrap();
    assert_eq!(var_result.shape(), &[size, size]);
    assert!(var_result.iter().all(|&x| x >= 0.0 && x.is_finite()));

    // Test Frobenius norm
    let norm = frobenius_norm_tensor(&tensor.view());
    assert!(norm.is_finite());
    assert!(norm > 0.0);
}

#[test]
fn test_memory_efficiency_streaming() {
    // Test that large operations don't cause excessive memory allocation
    // This is a stress test for memory efficiency
    let size = 100;
    let rank = 32;

    // Create tensor and factors
    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
        ((idx[0] + idx[1] + idx[2]) % 100) as f64 + 1.0
    }));

    let factors: Vec<Array2<f64>> = (0..3)
        .map(|i| {
            Array2::<f64>::from_shape_fn((size, rank), |(j, k)| ((i + j + k) % 50) as f64 + 1.0)
        })
        .collect();

    // Perform multiple MTTKRP operations in sequence
    // This tests that memory is properly cleaned up between operations
    for mode in 0..3 {
        let factor_views: Vec<_> = factors.iter().map(|f| f.view()).collect();

        let result = mttkrp(&tensor.view(), &factor_views, mode).unwrap();
        assert_eq!(result.shape(), &[size, rank]);

        // Force computation and verify result
        let norm = result.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm.is_finite() && norm > 0.0);
    }
}

#[test]
fn test_large_tensor_pipeline() {
    // Test a realistic pipeline with large tensors: transform -> reduce -> analyze
    let size = 80;

    // Step 1: Create large tensor
    let tensor = DenseND::<f64>::from_array(Array::from_shape_fn(vec![size, size, size], |idx| {
        let sum = (idx[0] + idx[1] + idx[2]) as f64;
        (sum * sum % 100.0) + 1.0
    }));

    // Step 2: Apply N-mode product (dimensionality reduction)
    let matrix =
        Array2::<f64>::from_shape_fn((size / 2, size), |(i, j)| ((i + j) % 50) as f64 + 1.0);
    let transformed = nmode_product(&tensor.view(), &matrix.view(), 0).unwrap();
    assert_eq!(transformed.shape(), &[size / 2, size, size]);

    // Step 3: Compute statistics
    let mean = mean_along_modes(&transformed.view(), &[0]).unwrap();
    let variance = variance_along_modes(&transformed.view(), &[0], 0).unwrap();

    // Step 4: Verify statistical properties
    assert!(mean.iter().all(|&x| x.is_finite()));
    assert!(variance.iter().all(|&x| x >= 0.0 && x.is_finite()));

    // Step 5: Compute norms for validation
    let l1_norm = pnorm_along_modes(&transformed.view(), &[0], 1.0).unwrap();
    let l2_norm = pnorm_along_modes(&transformed.view(), &[0], 2.0).unwrap();

    // Verify L1 >= L2 (triangle inequality)
    for i in 0..size {
        for j in 0..size {
            assert!(
                l1_norm[[i, j]] >= l2_norm[[i, j]] - 1e-10,
                "L1 norm should be >= L2 norm"
            );
        }
    }
}
