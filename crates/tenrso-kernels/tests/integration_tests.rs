//! Integration tests for tenrso-kernels with tenrso-core
//!
//! These tests verify that tensor operations work correctly when using
//! DenseND tensors from tenrso-core with kernel operations.

use scirs2_core::ndarray_ext::{s, Array2};
use tenrso_core::DenseND;
use tenrso_kernels::{hadamard, khatri_rao, kronecker, mttkrp, nmode_product};

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
