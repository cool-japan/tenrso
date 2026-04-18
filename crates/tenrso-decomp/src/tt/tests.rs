//! Unit tests for the TT module.
//!
//! These tests were originally co-located with `tt.rs`; they now live in a
//! dedicated file so the production sources stay well below the 2000-line
//! soft limit. `super::*` resolves against `tt/mod.rs`, which re-exports
//! every public symbol we rely on here.

// Bring the `Float` trait into scope so that method resolution on ambiguous
// numeric literals (e.g. `let mut max_diff = 0.0; max_diff.max(...)`) picks
// up the same signature as when the tests lived inside `tt.rs`.
#[allow(unused_imports)]
use scirs2_core::numeric::Float;

use super::*;
use tenrso_core::DenseND;

#[test]
fn test_tt_svd_basic() {
    // Small tensor for quick test
    let tensor = DenseND::<f64>::ones(&[3, 4, 5]);
    let result = tt_svd(&tensor, &[2, 2], 1e-10);

    if result.is_err() {
        eprintln!("TT-SVD error: {:?}", result.err());
        panic!("TT-SVD failed");
    }

    let tt = result.unwrap();

    assert_eq!(tt.cores.len(), 3);
    assert_eq!(tt.ranks.len(), 2);

    // Check core shapes
    assert_eq!(tt.cores[0].shape(), &[1, 3, tt.ranks[0]]);
    assert_eq!(tt.cores[1].shape(), &[tt.ranks[0], 4, tt.ranks[1]]);
    assert_eq!(tt.cores[2].shape(), &[tt.ranks[1], 5, 1]);
}

#[test]
fn test_tt_reconstruction() {
    let tensor = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
    let mut tt = tt_svd(&tensor, &[3, 4], 1e-10).unwrap();

    let reconstructed = tt.reconstruct();
    assert!(reconstructed.is_ok());

    let error = tt.compute_error(&tensor);
    assert!(error.is_ok());
    assert!(error.unwrap() < 0.5); // Reasonable reconstruction
}

#[test]
fn test_tt_compression() {
    let tensor = DenseND::<f64>::ones(&[10, 10, 10, 10]);
    let tt = tt_svd(&tensor, &[5, 5, 5], 1e-10).unwrap();

    let full_size = 10 * 10 * 10 * 10;
    let tt_size = tt.num_parameters();

    assert!(tt_size < full_size);
    assert!(tt.compression_ratio() > 1.0);
}

#[test]
fn test_tt_round_basic() {
    // Create a TT decomposition with smaller tensor for speed (4^4=256 vs 6^4=1296 elements)
    let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4, 4], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[3, 3, 3], 1e-10).unwrap();

    // Round to smaller ranks
    let tt_rounded = tt_round(&tt, &[2, 2, 2], 1e-6).unwrap();

    // Check that ranks are reduced
    for (i, &rank) in tt_rounded.ranks.iter().enumerate() {
        assert!(rank <= 2, "Rounded rank {} is {}, expected <= 2", i, rank);
    }

    // Check core shapes are valid
    assert_eq!(tt_rounded.cores.len(), 4);
    assert_eq!(tt_rounded.cores[0].shape()[0], 1); // First core left rank = 1
    assert_eq!(tt_rounded.cores[3].shape()[2], 1); // Last core right rank = 1
}

#[test]
fn test_tt_round_reconstruction() {
    // Create a low-rank tensor (rank-1 tensor = outer product)
    let tensor = DenseND::<f64>::random_uniform(&[5, 6, 7], 0.0, 1.0);

    // Decompose with larger ranks
    let tt = tt_svd(&tensor, &[4, 5], 1e-10).unwrap();

    // Round to smaller ranks
    let mut tt_rounded = tt_round(&tt, &[2, 2], 1e-6).unwrap();

    // Verify reconstruction is reasonable
    let reconstructed = tt_rounded.reconstruct().unwrap();
    assert_eq!(reconstructed.shape(), tensor.shape());

    let error = tt_rounded.compute_error(&tensor).unwrap();
    assert!(error < 1.0, "Reconstruction error too large: {}", error);
}

#[test]
fn test_tt_round_preserves_accuracy() {
    // Test that rounding with high tolerance preserves accuracy
    let tensor = DenseND::<f64>::random_uniform(&[6, 6, 6], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[5, 5], 1e-10).unwrap();

    // Round with very loose tolerance (should preserve accuracy)
    let mut tt_rounded = tt_round(&tt, &[5, 5], 1e-3).unwrap();

    let error = tt_rounded.compute_error(&tensor).unwrap();
    assert!(error < 0.3, "Error after rounding is too large: {}", error);
}

#[test]
fn test_tt_round_compression() {
    // Verify that rounding reduces storage (reduced size for speed)
    let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4, 4], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[3, 3, 3], 1e-10).unwrap();
    let original_params = tt.num_parameters();

    // Round to smaller ranks
    let tt_rounded = tt_round(&tt, &[2, 2, 2], 1e-6).unwrap();
    let rounded_params = tt_rounded.num_parameters();

    assert!(
        rounded_params < original_params,
        "Rounding should reduce parameters: {} >= {}",
        rounded_params,
        original_params
    );

    // Compression ratio should increase
    assert!(
        tt_rounded.compression_ratio() > tt.compression_ratio(),
        "Rounded compression ratio {} should be > original {}",
        tt_rounded.compression_ratio(),
        tt.compression_ratio()
    );
}

#[test]
fn test_tt_round_ranks_not_exceed_max() {
    // Test that rounded ranks never exceed max_ranks (reduced size for speed)
    let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4, 4], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[3, 3, 3], 1e-12).unwrap();

    let max_ranks = vec![2, 2, 2];
    let tt_rounded = tt_round(&tt, &max_ranks, 1e-8).unwrap();

    for (i, &rank) in tt_rounded.ranks.iter().enumerate() {
        assert!(
            rank <= max_ranks[i],
            "Rank {} is {}, exceeds max {}",
            i,
            rank,
            max_ranks[i]
        );
    }
}

// ========================================================================
// TT Operations Tests
// ========================================================================

#[test]
fn test_tt_add_basic() {
    // Test basic TT addition
    let tensor1 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
    let tensor2 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);

    let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
    let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

    let tt_sum = tt_add(&tt1, &tt2).unwrap();

    // Check shape is preserved
    assert_eq!(tt_sum.shape, vec![4, 5, 6]);

    // Check that ranks are sum of input ranks
    assert_eq!(tt_sum.ranks.len(), 2);
    for i in 0..tt_sum.ranks.len() {
        assert_eq!(
            tt_sum.ranks[i],
            tt1.ranks[i] + tt2.ranks[i],
            "Rank {} should be sum of input ranks",
            i
        );
    }

    // Verify reconstruction is approximately correct
    let recon_sum = tt_sum.reconstruct().unwrap();
    let recon1 = tt1.reconstruct().unwrap();
    let recon2 = tt2.reconstruct().unwrap();

    // Compute expected sum element-wise
    let mut expected_data = Vec::new();
    for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
        expected_data.push(v1 + v2);
    }
    let expected_sum = DenseND::from_vec(expected_data, &[4, 5, 6]).unwrap();

    // Compute difference
    let mut diff_data = Vec::new();
    for (vs, ve) in recon_sum.view().iter().zip(expected_sum.view().iter()) {
        diff_data.push(vs - ve);
    }
    let diff = DenseND::from_vec(diff_data, &[4, 5, 6]).unwrap();
    let relative_error = diff.frobenius_norm() / expected_sum.frobenius_norm();

    assert!(
        relative_error < 1e-6,
        "Addition reconstruction error too large: {}",
        relative_error
    );
}

#[test]
fn test_tt_dot_basic() {
    // Test TT inner product
    let tensor1 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);
    let tensor2 = DenseND::<f64>::random_uniform(&[5, 5, 5], 0.0, 1.0);

    let tt1 = tt_svd(&tensor1, &[4, 4], 1e-10).unwrap();
    let tt2 = tt_svd(&tensor2, &[4, 4], 1e-10).unwrap();

    // Compute inner product via TT
    let tt_inner_prod = tt_dot(&tt1, &tt2).unwrap();

    // Compute reference inner product via full reconstruction
    let recon1 = tt1.reconstruct().unwrap();
    let recon2 = tt2.reconstruct().unwrap();

    let mut expected_inner_prod = 0.0;
    for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
        expected_inner_prod += v1 * v2;
    }

    let relative_error = (tt_inner_prod - expected_inner_prod).abs() / expected_inner_prod.abs();

    assert!(
        relative_error < 1e-6,
        "Inner product error too large: TT={}, expected={}, rel_err={}",
        tt_inner_prod,
        expected_inner_prod,
        relative_error
    );
}

#[test]
fn test_tt_hadamard_basic() {
    // Test TT Hadamard product
    let tensor1 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
    let tensor2 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);

    let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
    let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

    let tt_prod = tt_hadamard(&tt1, &tt2).unwrap();

    // Check shape is preserved
    assert_eq!(tt_prod.shape, vec![4, 5, 6]);

    // Check that ranks are product of input ranks
    assert_eq!(tt_prod.ranks.len(), 2);
    for i in 0..tt_prod.ranks.len() {
        assert_eq!(
            tt_prod.ranks[i],
            tt1.ranks[i] * tt2.ranks[i],
            "Rank {} should be product of input ranks",
            i
        );
    }

    // Verify reconstruction is approximately correct
    let recon_prod = tt_prod.reconstruct().unwrap();
    let recon1 = tt1.reconstruct().unwrap();
    let recon2 = tt2.reconstruct().unwrap();

    // Compute expected element-wise product
    let mut expected_data = Vec::new();
    for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
        expected_data.push(v1 * v2);
    }
    let expected_prod = DenseND::from_vec(expected_data, &[4, 5, 6]).unwrap();

    // Compute difference
    let mut diff_data = Vec::new();
    for (vp, ve) in recon_prod.view().iter().zip(expected_prod.view().iter()) {
        diff_data.push(vp - ve);
    }
    let diff = DenseND::from_vec(diff_data, &[4, 5, 6]).unwrap();
    let relative_error = diff.frobenius_norm() / expected_prod.frobenius_norm();

    assert!(
        relative_error < 1e-6,
        "Hadamard product reconstruction error too large: {}",
        relative_error
    );
}

#[test]
fn test_tt_operations_shape_mismatch() {
    // Test that operations fail on shape mismatch
    let tensor1 = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
    let tensor2 = DenseND::<f64>::random_uniform(&[4, 5, 7], 0.0, 1.0); // Different last dimension

    let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
    let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

    // All operations should fail with shape mismatch
    assert!(tt_add(&tt1, &tt2).is_err());
    assert!(tt_dot(&tt1, &tt2).is_err());
    assert!(tt_hadamard(&tt1, &tt2).is_err());
}

#[test]
fn test_tt_add_with_rounding() {
    // Test that TT addition followed by rounding works correctly
    // Use larger tensors to avoid QR dimension issues
    let tensor1 = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);
    let tensor2 = DenseND::<f64>::random_uniform(&[8, 8, 8], 0.0, 1.0);

    let tt1 = tt_svd(&tensor1, &[3, 3], 1e-10).unwrap();
    let tt2 = tt_svd(&tensor2, &[3, 3], 1e-10).unwrap();

    // Add and round
    let tt_sum = tt_add(&tt1, &tt2).unwrap();
    let original_ranks = tt_sum.ranks.clone();

    let tt_rounded = tt_round(&tt_sum, &[4, 4], 1e-6).unwrap();

    // Rounded ranks should be <= original and <= max_ranks
    for (i, &rank) in tt_rounded.ranks.iter().enumerate() {
        assert!(rank <= original_ranks[i]);
        assert!(rank <= 4);
    }

    // Reconstruction should still be reasonable
    let recon_rounded = tt_rounded.reconstruct().unwrap();
    let recon1 = tt1.reconstruct().unwrap();
    let recon2 = tt2.reconstruct().unwrap();

    // Compute expected sum element-wise
    let mut expected_data = Vec::new();
    for (v1, v2) in recon1.view().iter().zip(recon2.view().iter()) {
        expected_data.push(v1 + v2);
    }
    let expected = DenseND::from_vec(expected_data, &[8, 8, 8]).unwrap();

    // Compute difference
    let mut diff_data = Vec::new();
    for (vr, ve) in recon_rounded.view().iter().zip(expected.view().iter()) {
        diff_data.push(vr - ve);
    }
    let diff = DenseND::from_vec(diff_data, &[8, 8, 8]).unwrap();
    let relative_error = diff.frobenius_norm() / expected.frobenius_norm();

    assert!(
        relative_error < 0.1,
        "Rounded sum error too large: {}",
        relative_error
    );
}

// ========================================================================
// TT Utility Methods Tests
// ========================================================================

#[test]
fn test_tt_eval_at() {
    // Create a small tensor for testing
    use scirs2_core::ndarray_ext::Array;
    let data = Array::from_shape_fn((3, 4, 5), |(i, j, k)| ((i + j + k) as f64) / 10.0);
    let tensor = DenseND::from_array(data.into_dyn());

    // Decompose
    let tt = tt_svd(&tensor, &[3, 3], 1e-10).unwrap();

    // Test evaluation at several indices
    let indices_list = vec![[0, 0, 0], [1, 2, 3], [2, 3, 4]];

    for indices in indices_list {
        // Get value via TT evaluation
        let tt_value = tt.eval_at(&indices).unwrap();

        // Get value from original tensor
        let orig_value = tensor.view()[[indices[0], indices[1], indices[2]]];

        // Should be very close (TT is approximate, so allow small error)
        let diff = (tt_value - orig_value).abs();
        // Use relative error for non-zero values, absolute error for zeros
        let tol = if orig_value.abs() > 1e-10 {
            orig_value.abs() * 0.01 // 1% relative error
        } else {
            1e-3 // Small absolute error for values close to zero
        };
        assert!(
            diff < tol,
            "TT eval_at mismatch at {:?}: TT={}, Orig={}, Diff={}, Tol={}",
            indices,
            tt_value,
            orig_value,
            diff,
            tol
        );
    }
}

#[test]
fn test_tt_eval_at_bounds_checking() {
    let tensor = DenseND::<f64>::random_uniform(&[3, 4, 5], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[2, 2], 1e-10).unwrap();

    // Test out of bounds indices
    assert!(tt.eval_at(&[3, 2, 2]).is_err()); // First index too large
    assert!(tt.eval_at(&[1, 4, 2]).is_err()); // Second index too large
    assert!(tt.eval_at(&[1, 2, 5]).is_err()); // Third index too large
    assert!(tt.eval_at(&[1, 2]).is_err()); // Too few indices
    assert!(tt.eval_at(&[1, 2, 3, 4]).is_err()); // Too many indices
}

#[test]
fn test_tt_frobenius_norm() {
    // Create a tensor and compute norm directly
    let tensor = DenseND::<f64>::random_uniform(&[5, 6, 7], 0.0, 1.0);
    let orig_norm = tensor.frobenius_norm();

    // Decompose and compute norm via TT with tight tolerance
    let tt = tt_svd(&tensor, &[4, 5], 1e-10).unwrap();
    let tt_norm = tt.frobenius_norm().unwrap();

    // Norms should be reasonably close (TT is approximate)
    // Allow 5% relative error due to truncation
    let diff = (tt_norm - orig_norm).abs();
    let relative_diff = diff / orig_norm;

    assert!(
        relative_diff < 0.05,
        "TT Frobenius norm differs from original: TT={}, Orig={}, Rel. Diff={}",
        tt_norm,
        orig_norm,
        relative_diff
    );
}

#[test]
fn test_tt_frobenius_norm_properties() {
    // Test that ||αX|| = |α| ||X||
    let tensor = DenseND::<f64>::random_uniform(&[4, 5, 6], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[3, 3], 1e-10).unwrap();
    let norm = tt.frobenius_norm().unwrap();

    // Scale all cores by 2.0
    let mut tt_scaled = tt.clone();
    for core in &mut tt_scaled.cores {
        *core = core.mapv(|x| x * 2.0);
    }

    let norm_scaled = tt_scaled.frobenius_norm().unwrap();

    // Should be approximately 2^n_modes * norm (since we scaled each core)
    // Actually, the scaling propagates through all cores
    let expected_factor = 2.0_f64.powi(tt.cores.len() as i32);
    let expected_norm = norm * expected_factor;

    let rel_diff = (norm_scaled - expected_norm).abs() / expected_norm;
    assert!(
        rel_diff < 1e-6,
        "Scaled norm incorrect: got {}, expected {}",
        norm_scaled,
        expected_norm
    );
}

#[test]
fn test_tt_max_rank() {
    // Reduced to [4,4,4] for speed (64 elements)
    let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[3, 2], 1e-8).unwrap();

    let max_rank = tt.max_rank();
    assert_eq!(max_rank, 3); // max of [3, 2]
}

#[test]
fn test_tt_effective_rank() {
    // Reduced to [4,4,4] for speed (64 elements)
    let tensor = DenseND::<f64>::random_uniform(&[4, 4, 4], 0.0, 1.0);
    let tt = tt_svd(&tensor, &[3, 2], 1e-8).unwrap();

    let eff_rank = tt.effective_rank();

    // Effective rank should be average of actual ranks
    let n = tt.ranks.len();
    let sum: usize = tt.ranks.iter().sum();
    let expected = sum as f64 / n as f64;
    assert!((eff_rank - expected).abs() < 1e-10);

    // Should be between min and max rank
    let min_rank = *tt.ranks.iter().min().unwrap() as f64;
    let max_rank = *tt.ranks.iter().max().unwrap() as f64;
    assert!(eff_rank >= min_rank);
    assert!(eff_rank <= max_rank);
}

#[test]
fn test_tt_matrix_from_diagonal() {
    use scirs2_core::ndarray_ext::Array1;

    // Create diagonal matrix
    let diag = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let tt_mat = tt_matrix_from_diagonal(&diag, &[2, 2, 2]);

    assert_eq!(tt_mat.cores.len(), 3);
    assert_eq!(tt_mat.out_shape, vec![2, 2, 2]);
    assert_eq!(tt_mat.in_shape, vec![2, 2, 2]);
    assert_eq!(tt_mat.ranks, vec![1, 1]);

    // Check core shapes: (r_{k-1}, n_k, m_k, r_k)
    assert_eq!(tt_mat.cores[0].shape(), [1, 2, 2, 1]);
    assert_eq!(tt_mat.cores[1].shape(), [1, 2, 2, 1]);
    assert_eq!(tt_mat.cores[2].shape(), [1, 2, 2, 1]);
}

#[test]
fn test_tt_matvec_basic() {
    use scirs2_core::ndarray_ext::Array1;

    // Create identity matrix in TT format (diagonal with all 1s)
    let ones = Array1::from_vec(vec![1.0; 8]);
    let identity = tt_matrix_from_diagonal(&ones, &[2, 2, 2]);

    // Create a vector in TT format
    let vec = DenseND::<f64>::random_uniform(&[2, 2, 2], 0.0, 1.0);
    let tt_vec = tt_svd(&vec, &[2, 2], 1e-10).unwrap();

    // Multiply: identity × vec should equal vec
    let result = identity.matvec(&tt_vec).unwrap();

    // Reconstruct both and compare
    let vec_reconstructed = tt_vec.reconstruct().unwrap();
    let result_reconstructed = result.reconstruct().unwrap();

    // Check shapes match
    assert_eq!(result_reconstructed.shape(), vec_reconstructed.shape());

    // Values should be close (up to numerical error)
    let vec_view = vec_reconstructed.view();
    let res_view = result_reconstructed.view();

    let mut max_diff = 0.0;
    for (v1, v2) in vec_view.iter().zip(res_view.iter()) {
        max_diff = max_diff.max((v1 - v2).abs());
    }

    assert!(max_diff < 1e-10, "Max difference: {}", max_diff);
}

#[test]
fn test_tt_matvec_scaling() {
    use scirs2_core::ndarray_ext::Array1;

    // Create scaling matrix: diag([1, 2, 3, 4, 5, 6, 7, 8])
    let scale_factors = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let scale_matrix = tt_matrix_from_diagonal(&scale_factors, &[2, 2, 2]);

    // Create a known vector (all ones)
    let data = scirs2_core::ndarray_ext::Array::from_elem(
        scirs2_core::ndarray_ext::IxDyn(&[2, 2, 2]),
        1.0,
    );
    let vec = DenseND::<f64>::from_array(data);
    let tt_vec = tt_svd(&vec, &[2, 2], 1e-10).unwrap();

    // Multiply
    let result = scale_matrix.matvec(&tt_vec).unwrap();
    let result_dense = result.reconstruct().unwrap();

    // Check result dimensions
    assert_eq!(result_dense.shape(), &[2, 2, 2]);

    // Result should have bounded values (scaled by diagonal elements)
    let result_view = result_dense.view();
    let min_val = result_view.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = result_view
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    assert!(min_val >= 0.0, "Min value: {}", min_val);
    assert!(max_val <= 10.0, "Max value: {}", max_val);
}

#[test]
fn test_tt_matvec_dimension_mismatch() {
    use scirs2_core::ndarray_ext::Array1;

    // Create matrix for shape [2, 2, 2]
    let diag1 = Array1::from_vec(vec![1.0; 8]);
    let matrix = tt_matrix_from_diagonal(&diag1, &[2, 2, 2]);

    // Create vector for shape [2, 3]
    let vec = DenseND::<f64>::random_uniform(&[2, 3], 0.0, 1.0);
    let tt_vec = tt_svd(&vec, &[3], 1e-10).unwrap();

    // Should error due to dimension mismatch
    let result = matrix.matvec(&tt_vec);
    assert!(result.is_err());
}

#[test]
fn test_tt_matvec_ranks() {
    use scirs2_core::ndarray_ext::Array1;

    // Create diagonal matrix (rank-1)
    let diag = Array1::from_vec(vec![2.0; 16]);
    let matrix = tt_matrix_from_diagonal(&diag, &[2, 2, 2, 2]);

    // Create vector with higher ranks
    let vec = DenseND::<f64>::random_uniform(&[2, 2, 2, 2], 0.0, 1.0);
    let tt_vec = tt_svd(&vec, &[3, 3, 3], 1e-10).unwrap();

    // Result ranks should be product of matrix and vector ranks
    let result = matrix.matvec(&tt_vec).unwrap();

    // Matrix has all ranks = 1, vector has ranks [3, 3, 3]
    // Result should have ranks = [1*3, 1*3, 1*3] = [3, 3, 3]
    for (i, &rank) in result.ranks.iter().enumerate() {
        assert!(rank <= 3, "Rank {} is {}, expected <= 3", i, rank);
    }
}
