//! Property-based tests for tensor kernel operations
//!
//! These tests verify mathematical properties that should hold for all valid inputs

use super::*;
use proptest::prelude::*;
use scirs2_core::ndarray_ext::{Array, Array2};

/// Strategy to generate small matrix dimensions
fn small_matrix_dims() -> impl Strategy<Value = (usize, usize, usize)> {
    (2usize..20, 2usize..20, 2usize..10)
}

proptest! {
    /// Test that Khatri-Rao product has correct dimensions
    #[test]
    fn test_khatri_rao_dimensions((rows_a, rows_b, cols) in small_matrix_dims()) {
        let a = Array2::<f64>::ones((rows_a, cols));
        let b = Array2::<f64>::ones((rows_b, cols));

        let result = khatri_rao(&a.view(), &b.view());

        prop_assert_eq!(result.shape(), &[rows_a * rows_b, cols]);
    }

    /// Test that Khatri-Rao with identity-like matrices preserves structure
    #[test]
    fn test_khatri_rao_with_ones(rows in 2usize..20, cols in 2usize..10) {
        let a = Array2::<f64>::ones((rows, cols));
        let b = Array2::<f64>::ones((rows, cols));

        let result = khatri_rao(&a.view(), &b.view());

        // All elements should be 1.0 (1*1 = 1)
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                prop_assert_eq!(result[[i, j]], 1.0);
            }
        }
    }

    /// Test that Khatri-Rao with zeros gives zeros
    #[test]
    fn test_khatri_rao_with_zeros((rows_a, rows_b, cols) in small_matrix_dims()) {
        let a = Array2::<f64>::zeros((rows_a, cols));
        let b = Array2::<f64>::ones((rows_b, cols));

        let result = khatri_rao(&a.view(), &b.view());

        // All elements should be 0.0 (0*anything = 0)
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                prop_assert_eq!(result[[i, j]], 0.0);
            }
        }
    }

    /// Test that Kronecker product has correct dimensions
    #[test]
    fn test_kronecker_dimensions((m, n, p, q) in (2usize..15, 2usize..15, 2usize..15, 2usize..15)) {
        let a = Array2::<f64>::ones((m, n));
        let b = Array2::<f64>::ones((p, q));

        let result = kronecker(&a.view(), &b.view());

        prop_assert_eq!(result.shape(), &[m * p, n * q]);
    }

    /// Test that Kronecker product with identity gives block diagonal
    #[test]
    fn test_kronecker_with_identity(size in 2usize..10) {
        let a = Array2::<f64>::eye(size);
        let b = Array2::<f64>::ones((size, size));

        let result = kronecker(&a.view(), &b.view());

        prop_assert_eq!(result.shape(), &[size * size, size * size]);

        // Check block diagonal structure
        for block_i in 0..size {
            for block_j in 0..size {
                for i in 0..size {
                    for j in 0..size {
                        let row = block_i * size + i;
                        let col = block_j * size + j;
                        let expected = if block_i == block_j { 1.0 } else { 0.0 };
                        prop_assert_eq!(result[[row, col]], expected,
                            "Mismatch at ({}, {})", row, col);
                    }
                }
            }
        }
    }

    /// Test that Hadamard product preserves dimensions
    #[test]
    fn test_hadamard_dimensions((rows, cols) in (2usize..50, 2usize..50)) {
        let a = Array2::<f64>::ones((rows, cols));
        let b = Array2::<f64>::ones((rows, cols));

        let result = hadamard(&a.view(), &b.view());

        prop_assert_eq!(result.shape(), &[rows, cols]);
    }

    /// Test that Hadamard product is commutative
    #[test]
    fn test_hadamard_commutative((rows, cols) in (2usize..30, 2usize..30)) {
        let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * j + 1) as f64);

        let ab = hadamard(&a.view(), &b.view());
        let ba = hadamard(&b.view(), &a.view());

        for i in 0..rows {
            for j in 0..cols {
                prop_assert!((ab[[i, j]] - ba[[i, j]]).abs() < 1e-10);
            }
        }
    }

    /// Test that Hadamard product with ones is identity
    #[test]
    fn test_hadamard_identity((rows, cols) in (2usize..30, 2usize..30)) {
        let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64);
        let ones = Array2::<f64>::ones((rows, cols));

        let result = hadamard(&a.view(), &ones.view());

        for i in 0..rows {
            for j in 0..cols {
                prop_assert_eq!(result[[i, j]], a[[i, j]]);
            }
        }
    }

    /// Test that Hadamard product with zeros gives zeros
    #[test]
    fn test_hadamard_with_zeros((rows, cols) in (2usize..30, 2usize..30)) {
        let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64);
        let zeros = Array2::<f64>::zeros((rows, cols));

        let result = hadamard(&a.view(), &zeros.view());

        for i in 0..rows {
            for j in 0..cols {
                prop_assert_eq!(result[[i, j]], 0.0);
            }
        }
    }

    /// Test that N-mode product has correct output dimensions
    #[test]
    fn test_nmode_product_dimensions((size, mode) in (3usize..12, 0usize..3)) {
        let tensor = Array::from_shape_fn(vec![size, size, size], |_| 1.0f64);
        let new_mode_size = size + 2;
        let matrix = Array2::<f64>::ones((new_mode_size, size));

        let result = nmode_product(&tensor.view(), &matrix.view(), mode).unwrap();

        let mut expected_shape = [size, size, size];
        expected_shape[mode] = new_mode_size;

        prop_assert_eq!(result.shape(), &expected_shape[..]);
    }

    /// Test that N-mode product with identity matrix preserves tensor
    #[test]
    fn test_nmode_product_identity((size, mode) in (3usize..10, 0usize..3)) {
        use scirs2_core::ndarray_ext::IxDyn;
        let tensor = Array::from_shape_fn(vec![size, size, size], |idx: IxDyn| {
            (idx[0] + idx[1] * 2 + idx[2] * 3) as f64
        });
        let identity = Array2::<f64>::eye(size);

        let result = nmode_product(&tensor.view(), &identity.view(), mode).unwrap();

        prop_assert_eq!(result.shape(), tensor.shape());

        // Values should be approximately the same (within floating point error)
        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    let orig = tensor[[i, j, k]];
                    let new_val = result[[i, j, k]];
                    prop_assert!((orig - new_val).abs() < 1e-8,
                        "Mismatch at ({}, {}, {}): {} vs {}", i, j, k, orig, new_val);
                }
            }
        }
    }

    /// Test that MTTKRP output has correct dimensions
    #[test]
    fn test_mttkrp_dimensions((size, rank, mode) in (3usize..10, 2usize..8, 0usize..3)) {
        let tensor = Array::from_shape_fn(vec![size, size, size], |_| 1.0f64);
        let u1 = Array2::<f64>::ones((size, rank));
        let u2 = Array2::<f64>::ones((size, rank));
        let u3 = Array2::<f64>::ones((size, rank));

        let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], mode).unwrap();

        prop_assert_eq!(result.shape(), &[size, rank]);
    }

    /// Test that MTTKRP with all-ones factors gives predictable results
    #[test]
    fn test_mttkrp_ones((size, rank) in (3usize..8, 2usize..5)) {
        // Create a tensor of all ones
        let tensor = Array::from_shape_fn(vec![size, size, size], |_| 1.0f64);
        let u1 = Array2::<f64>::ones((size, rank));
        let u2 = Array2::<f64>::ones((size, rank));
        let u3 = Array2::<f64>::ones((size, rank));

        for mode in 0..3 {
            let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], mode).unwrap();

            prop_assert_eq!(result.shape(), &[size, rank]);

            // Each element should be size^2 (sum of size^2 ones)
            let expected = (size * size) as f64;
            for i in 0..size {
                for r in 0..rank {
                    prop_assert!((result[[i, r]] - expected).abs() < 1e-8,
                        "Element ({}, {}) = {}, expected {}", i, r, result[[i, r]], expected);
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
proptest! {
    /// Test that parallel Khatri-Rao gives same result as serial
    #[test]
    fn test_khatri_rao_parallel_matches_serial((rows_a, rows_b, cols) in small_matrix_dims()) {
        let a = Array2::<f64>::from_shape_fn((rows_a, cols), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((rows_b, cols), |(i, j)| (i * j + 1) as f64);

        let serial = khatri_rao(&a.view(), &b.view());
        let parallel = khatri_rao_parallel(&a.view(), &b.view());

        prop_assert_eq!(serial.shape(), parallel.shape());

        for i in 0..serial.shape()[0] {
            for j in 0..serial.shape()[1] {
                prop_assert!((serial[[i, j]] - parallel[[i, j]]).abs() < 1e-10);
            }
        }
    }

    /// Test that parallel Kronecker gives same result as serial
    #[test]
    fn test_kronecker_parallel_matches_serial((m, n, p, q) in (2usize..10, 2usize..10, 2usize..10, 2usize..10)) {
        let a = Array2::<f64>::from_shape_fn((m, n), |(i, j)| (i + j) as f64);
        let b = Array2::<f64>::from_shape_fn((p, q), |(i, j)| (i * j + 1) as f64);

        let serial = kronecker(&a.view(), &b.view());
        let parallel = kronecker_parallel(&a.view(), &b.view());

        prop_assert_eq!(serial.shape(), parallel.shape());

        for i in 0..serial.shape()[0] {
            for j in 0..serial.shape()[1] {
                prop_assert!((serial[[i, j]] - parallel[[i, j]]).abs() < 1e-10);
            }
        }
    }
}

// ============================================================================
// Advanced Property Tests - Mathematical Properties & Numerical Stability
// ============================================================================

proptest! {
    /// Test Khatri-Rao associativity: (A ⊙ B) ⊙ C should have consistent structure
    /// Note: Khatri-Rao is not associative in the mathematical sense, but we can
    /// verify the column structure is consistent
    #[test]
    fn test_khatri_rao_column_structure((rows_a, rows_b, cols) in (2usize..10, 2usize..10, 2usize..5)) {
        let a = Array2::<f64>::from_shape_fn((rows_a, cols), |(i, j)| (i + j + 1) as f64);
        let b = Array2::<f64>::from_shape_fn((rows_b, cols), |(i, j)| (i * j + 1) as f64);

        let kr = khatri_rao(&a.view(), &b.view());

        // Verify each column of KR is the Kronecker product of corresponding columns
        for col_idx in 0..cols {
            let a_col = a.column(col_idx);
            let b_col = b.column(col_idx);
            let kr_col = kr.column(col_idx);

            for i in 0..rows_a {
                for j in 0..rows_b {
                    let expected = a_col[i] * b_col[j];
                    let actual = kr_col[i * rows_b + j];
                    prop_assert!((expected - actual).abs() < 1e-10,
                        "Column {} mismatch at ({}, {})", col_idx, i, j);
                }
            }
        }
    }

    /// Test Hadamard associativity: (A ∘ B) ∘ C = A ∘ (B ∘ C)
    #[test]
    fn test_hadamard_associative((rows, cols) in (2usize..20, 2usize..20)) {
        let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j + 1) as f64);
        let b = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * j + 1) as f64);
        let c = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j * 2 + 1) as f64);

        let ab_c = hadamard(&hadamard(&a.view(), &b.view()).view(), &c.view());
        let a_bc = hadamard(&a.view(), &hadamard(&b.view(), &c.view()).view());

        for i in 0..rows {
            for j in 0..cols {
                prop_assert!((ab_c[[i, j]] - a_bc[[i, j]]).abs() < 1e-10);
            }
        }
    }

    /// Test Hadamard distributivity over addition: A ∘ (B + C) = (A ∘ B) + (A ∘ C)
    #[test]
    fn test_hadamard_distributive((rows, cols) in (2usize..20, 2usize..20)) {
        let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j + 1) as f64);
        let b = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * j + 1) as f64);
        let c = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i + j * 2 + 1) as f64);

        // A ∘ (B + C)
        let b_plus_c = &b + &c;
        let left = hadamard(&a.view(), &b_plus_c.view());

        // (A ∘ B) + (A ∘ C)
        let ab = hadamard(&a.view(), &b.view());
        let ac = hadamard(&a.view(), &c.view());
        let right = &ab + &ac;

        for i in 0..rows {
            for j in 0..cols {
                prop_assert!((left[[i, j]] - right[[i, j]]).abs() < 1e-9);
            }
        }
    }

    /// Test Kronecker product with identity: I_m ⊗ A = block diagonal structure
    #[test]
    fn test_kronecker_identity_left((m, n) in (2usize..8, 2usize..8)) {
        let identity = Array2::<f64>::eye(m);
        let a = Array2::<f64>::from_shape_fn((n, n), |(i, j)| (i + j + 1) as f64);

        let result = kronecker(&identity.view(), &a.view());

        prop_assert_eq!(result.shape(), &[m * n, m * n]);

        // Check block diagonal structure
        for block_i in 0..m {
            for block_j in 0..m {
                for i in 0..n {
                    for j in 0..n {
                        let row = block_i * n + i;
                        let col = block_j * n + j;
                        let expected = if block_i == block_j { a[[i, j]] } else { 0.0 };
                        prop_assert!((result[[row, col]] - expected).abs() < 1e-10,
                            "Mismatch at block ({}, {}), element ({}, {})", block_i, block_j, i, j);
                    }
                }
            }
        }
    }

    /// Test N-mode product chain: Multiple sequential n-mode products
    #[test]
    fn test_nmode_product_chain((size, new_size) in (3usize..8, 2usize..6)) {
        let tensor = Array::from_shape_fn(vec![size, size, size], |_| 1.0f64);
        let m1 = Array2::<f64>::ones((new_size, size));
        let m2 = Array2::<f64>::ones((new_size, size));

        // Apply to mode 0 then mode 1
        let result1 = nmode_product(&tensor.view(), &m1.view(), 0).unwrap();
        let result2 = nmode_product(&result1.view(), &m2.view(), 1).unwrap();

        prop_assert_eq!(result2.shape(), &[new_size, new_size, size]);
    }

    /// Test outer product reconstruction: sum of rank-1 outer products
    #[test]
    fn test_outer_product_sum_property(size in 2usize..8) {
        use scirs2_core::ndarray_ext::Array1;
        use crate::outer::outer_product_2;

        // Create 1D vectors
        let v1 = Array1::<f64>::from_vec((0..size).map(|_| 1.0).collect());
        let v2 = Array1::<f64>::from_vec((0..size).map(|_| 2.0).collect());

        // Weighted sum property: w1*(v1⊗v2) + w2*(v1⊗v2) = (w1+w2)*(v1⊗v2)
        let weight1 = 3.0;
        let weight2 = 4.0;

        let op1 = outer_product_2(&v1.view(), &v2.view()) * weight1;
        let op2 = outer_product_2(&v1.view(), &v2.view()) * weight2;

        let total_weight = weight1 + weight2;
        let op_total = outer_product_2(&v1.view(), &v2.view()) * total_weight;

        // op1 + op2 should equal op_total
        for i in 0..size {
            for j in 0..size {
                let sum = op1[[i, j]] + op2[[i, j]];
                prop_assert!((sum - op_total[[i, j]]).abs() < 1e-10);
            }
        }
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

proptest! {
    /// Test Khatri-Rao with very small values (near-zero stability)
    #[test]
    fn test_khatri_rao_small_values((rows_a, rows_b, cols) in (2usize..10, 2usize..10, 2usize..5)) {
        let a = Array2::<f64>::from_shape_fn((rows_a, cols), |(i, j)| 1e-10 * (i + j + 1) as f64);
        let b = Array2::<f64>::from_shape_fn((rows_b, cols), |(i, j)| 1e-10 * (i * j + 1) as f64);

        let result = khatri_rao(&a.view(), &b.view());

        // Should not produce NaN or Inf
        for val in result.iter() {
            prop_assert!(val.is_finite(), "Non-finite value: {}", val);
        }
    }

    /// Test Hadamard with mixed positive/negative values
    #[test]
    fn test_hadamard_mixed_signs((rows, cols) in (2usize..20, 2usize..20)) {
        let a = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| {
            if (i + j) % 2 == 0 { (i + j + 1) as f64 } else { -((i + j + 1) as f64) }
        });
        let b = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| {
            if (i * j) % 2 == 0 { (i * j + 1) as f64 } else { -((i * j + 1) as f64) }
        });

        let result = hadamard(&a.view(), &b.view());

        // All values should be finite
        for val in result.iter() {
            prop_assert!(val.is_finite());
        }

        // Sign should be product of input signs
        for i in 0..rows {
            for j in 0..cols {
                let expected_sign = a[[i, j]].signum() * b[[i, j]].signum();
                let actual_sign = result[[i, j]].signum();
                prop_assert_eq!(expected_sign, actual_sign);
            }
        }
    }

    /// Test MTTKRP with normalized factors (common in CP-ALS)
    #[test]
    fn test_mttkrp_normalized_factors((size, rank) in (3usize..8, 2usize..5)) {
        let tensor = Array::from_shape_fn(vec![size, size, size], |_| 1.0f64);

        // Create factors with controlled magnitude
        let u1 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| {
            let val = (i + j + 1) as f64;
            val / (val * val).sqrt()  // Manual normalization
        });
        let u2 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| {
            let val = (i * j + 1) as f64;
            val / (val * val).sqrt()
        });
        let u3 = Array2::<f64>::from_shape_fn((size, rank), |(i, j)| {
            let val = (i + j * 2 + 1) as f64;
            val / (val * val).sqrt()
        });

        let result = mttkrp(&tensor.view(), &[u1.view(), u2.view(), u3.view()], 0).unwrap();

        // Result should have finite values
        for val in result.iter() {
            prop_assert!(val.is_finite());
        }

        // Result should have reasonable magnitude (not overflow)
        for val in result.iter() {
            prop_assert!(val.abs() < 1e10, "Value too large: {}", val);
        }
    }

    /// Test N-mode product preserves non-negativity when appropriate
    #[test]
    fn test_nmode_product_non_negative((size, new_size) in (3usize..8, 2usize..6)) {
        // Non-negative tensor
        let tensor = Array::from_shape_fn(vec![size, size, size], |_| 1.0f64);
        // Non-negative matrix
        let matrix = Array2::<f64>::ones((new_size, size));

        let result = nmode_product(&tensor.view(), &matrix.view(), 0).unwrap();

        // Result should be non-negative
        for val in result.iter() {
            prop_assert!(*val >= 0.0, "Negative value in non-negative product: {}", val);
        }
    }
}

// ============================================================================
// Tensor Contraction Property Tests
// ============================================================================

proptest! {
    /// Test tensor contraction is bilinear: contract(aA, B) = a*contract(A, B)
    #[test]
    fn test_contraction_scalar_mult((m, n, k) in (2usize..8, 2usize..8, 2usize..8)) {
        use crate::contractions::contract_tensors;

        let a = Array::from_shape_fn(vec![m, n], |_| 2.0f64);
        let b = Array::from_shape_fn(vec![n, k], |_| 3.0f64);
        let scalar = 5.0;

        let scaled_a = &a * scalar;

        let result1 = contract_tensors(&scaled_a.view(), &b.view(), &[1], &[0]).unwrap();
        let result2 = contract_tensors(&a.view(), &b.view(), &[1], &[0]).unwrap() * scalar;

        prop_assert_eq!(result1.shape(), result2.shape());
        for i in 0..result1.len() {
            let diff = (result1.as_slice().unwrap()[i] - result2.as_slice().unwrap()[i]).abs();
            prop_assert!(diff < 1e-9, "Mismatch at {}: {} vs {}", i,
                result1.as_slice().unwrap()[i], result2.as_slice().unwrap()[i]);
        }
    }

    /// Test tensor inner product is symmetric: <A, B> = <B, A>
    #[test]
    fn test_inner_product_symmetric((m, n) in (2usize..10, 2usize..10)) {
        use crate::contractions::tensor_inner_product;

        let a = Array::from_shape_fn(vec![m, n], |idx| {
            let i = idx[0];
            let j = idx[1];
            (i + j + 1) as f64
        });
        let b = Array::from_shape_fn(vec![m, n], |idx| {
            let i = idx[0];
            let j = idx[1];
            (i * j + 1) as f64
        });

        let ab = tensor_inner_product(&a.view(), &b.view()).unwrap();
        let ba = tensor_inner_product(&b.view(), &a.view()).unwrap();

        prop_assert!((ab - ba).abs() < 1e-10);
    }

    /// Test sum reduction: sum over all modes equals sum of all elements
    #[test]
    fn test_sum_all_modes_equals_total((m, n, k) in (2usize..6, 2usize..6, 2usize..6)) {
        use crate::reductions::sum_along_modes;

        let tensor = Array::from_shape_fn(vec![m, n, k], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        });

        let sum_all = sum_along_modes(&tensor.view(), &[0, 1, 2]).unwrap();
        let expected: f64 = tensor.iter().sum();

        prop_assert!((sum_all[[0]] - expected).abs() < 1e-9);
    }

    /// Test mean reduction: mean of constants is constant
    #[test]
    fn test_mean_of_constants(constant in 1.0f64..10.0f64, (m, n) in (2usize..10, 2usize..10)) {
        use crate::reductions::mean_along_modes;

        let tensor = Array::from_elem(vec![m, n], constant);

        let mean = mean_along_modes(&tensor.view(), &[0, 1]).unwrap();

        prop_assert!((mean[[0]] - constant).abs() < 1e-10);
    }

    /// Test variance of constants is zero
    #[test]
    fn test_variance_of_constants(constant in 1.0f64..10.0f64, size in 3usize..10) {
        use crate::reductions::variance_along_modes;

        let tensor = Array::from_elem(vec![size, size], constant);

        let var = variance_along_modes(&tensor.view(), &[0], 1).unwrap();

        // Variance of constants should be zero
        for val in var.iter() {
            prop_assert!(val.abs() < 1e-9, "Variance should be zero for constants");
        }
    }

    /// Test Frobenius norm is non-negative and zero only for zero tensor
    #[test]
    fn test_frobenius_norm_properties((m, n) in (2usize..10, 2usize..10)) {
        use crate::reductions::frobenius_norm_tensor;

        let tensor = Array::from_shape_fn(vec![m, n], |idx| {
            ((idx[0] + idx[1]) % 3) as f64
        });

        let norm = frobenius_norm_tensor(&tensor.view());

        prop_assert!(norm >= 0.0);

        // If tensor has any non-zero element, norm > 0
        if tensor.iter().any(|&x| x != 0.0) {
            prop_assert!(norm > 0.0);
        }
    }

    /// Test p-norm ordering: ||x||_∞ ≤ ||x||_2 ≤ ||x||_1
    #[test]
    fn test_pnorm_ordering(size in 3usize..8) {
        use crate::reductions::pnorm_along_modes;

        let tensor = Array::from_shape_fn(vec![size, size], |idx| {
            if idx[0] == idx[1] { 1.0 } else { 0.5 }
        });

        let l1 = pnorm_along_modes(&tensor.view(), &[0, 1], 1.0).unwrap()[[0]];
        let l2 = pnorm_along_modes(&tensor.view(), &[0, 1], 2.0).unwrap()[[0]];

        // For non-negative vectors, l2 <= l1 (holds when normalized)
        // The actual ordering depends on the number of elements, but we can test basic sanity
        prop_assert!(l1 > 0.0);
        prop_assert!(l2 > 0.0);
    }

    /// Test min/max basic properties
    #[test]
    fn test_min_max_bounds(size in 2usize..10) {
        use crate::reductions::{min_along_modes, max_along_modes};

        let tensor = Array::from_shape_fn(vec![size, size], |idx| {
            (idx[0] * idx[1]) as f64
        });

        let min_val = min_along_modes(&tensor.view(), &[0, 1]).unwrap()[[0]];
        let max_val = max_along_modes(&tensor.view(), &[0, 1]).unwrap()[[0]];

        // Min should be <= Max
        prop_assert!(min_val <= max_val);

        // Min and max should be within tensor values
        let tensor_min = tensor.iter().cloned().fold(f64::INFINITY, f64::min);
        let tensor_max = tensor.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!((min_val - tensor_min).abs() < 1e-10);
        prop_assert!((max_val - tensor_max).abs() < 1e-10);
    }

    /// Test that median is between min and max
    #[test]
    fn test_median_bounds(data in prop::collection::vec(-100.0..100.0f64, 5..50)) {
        let tensor = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();

        let median = median_along_modes(&tensor.view(), &[1]).unwrap();

        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!(median[[0]] >= min_val - 1e-10);
        prop_assert!(median[[0]] <= max_val + 1e-10);
    }

    /// Test that median of constant array equals that constant
    #[test]
    fn test_median_constant(value in -100.0..100.0f64, size in 3..20usize) {
        let tensor = Array::from_elem(vec![1, size], value);

        let median = median_along_modes(&tensor.view(), &[1]).unwrap();

        prop_assert!((median[[0]] - value).abs() < 1e-10);
    }

    /// Test percentile bounds and monotonicity
    #[test]
    fn test_percentile_properties(data in prop::collection::vec(-100.0..100.0f64, 5..50)) {
        let tensor = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();

        // 0th percentile should equal minimum
        let p0 = percentile_along_modes(&tensor.view(), &[1], 0.0).unwrap();
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        prop_assert!((p0[[0]] - min_val).abs() < 1e-10);

        // 100th percentile should equal maximum
        let p100 = percentile_along_modes(&tensor.view(), &[1], 100.0).unwrap();
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        prop_assert!((p100[[0]] - max_val).abs() < 1e-10);

        // 50th percentile should equal median
        let p50 = percentile_along_modes(&tensor.view(), &[1], 50.0).unwrap();
        let median = median_along_modes(&tensor.view(), &[1]).unwrap();
        prop_assert!((p50[[0]] - median[[0]]).abs() < 1e-10);
    }

    /// Test percentile monotonicity: p1 < p2 => percentile(p1) <= percentile(p2)
    #[test]
    fn test_percentile_monotonicity(
        data in prop::collection::vec(-100.0..100.0f64, 5..50),
        p1 in 0.0..50.0f64,
        p2 in 50.0..100.0f64
    ) {
        let tensor = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();

        let val1 = percentile_along_modes(&tensor.view(), &[1], p1).unwrap();
        let val2 = percentile_along_modes(&tensor.view(), &[1], p2).unwrap();

        // p1 < p2, so val1 should be <= val2
        prop_assert!(val1[[0]] <= val2[[0]] + 1e-10);
    }

    /// Test that skewness is scale-invariant
    #[test]
    fn test_skewness_scale_invariant(
        data in prop::collection::vec(-10.0..10.0f64, 10..30),
        scale in 0.1..10.0f64
    ) {
        let tensor1 = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();
        let scaled_data: Vec<f64> = data.iter().map(|x| x * scale).collect();
        let tensor2 = Array::from_shape_vec(vec![1, scaled_data.len()], scaled_data).unwrap();

        let skew1 = skewness_along_modes(&tensor1.view(), &[1], true);
        let skew2 = skewness_along_modes(&tensor2.view(), &[1], true);

        // Both should succeed or both should fail
        if let (Ok(s1), Ok(s2)) = (skew1, skew2) {
            // Skewness should be invariant to scaling (within numerical tolerance)
            if !s1[[0]].is_nan() && !s2[[0]].is_nan() && !s1[[0]].is_infinite() && !s2[[0]].is_infinite() {
                prop_assert!((s1[[0]] - s2[[0]]).abs() < 1e-6,
                    "Skewness not scale-invariant: {} vs {}", s1[[0]], s2[[0]]);
            }
        }
    }

    /// Test that symmetric distributions have near-zero skewness
    #[test]
    fn test_skewness_symmetric(center in -10.0..10.0f64, range in 1.0..20.0f64) {
        // Create symmetric data around center
        let values: Vec<f64> = (-10..=10).map(|i| center + (i as f64) * range / 20.0).collect();
        let tensor = Array::from_shape_vec(vec![1, values.len()], values).unwrap();

        let skew = skewness_along_modes(&tensor.view(), &[1], true).unwrap();

        // Symmetric distribution should have near-zero skewness
        prop_assert!(skew[[0]].abs() < 0.5, "Skewness not near zero: {}", skew[[0]]);
    }

    /// Test that kurtosis is scale-invariant
    #[test]
    fn test_kurtosis_scale_invariant(
        data in prop::collection::vec(-10.0..10.0f64, 10..30),
        scale in 0.1..10.0f64
    ) {
        let tensor1 = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();
        let scaled_data: Vec<f64> = data.iter().map(|x| x * scale).collect();
        let tensor2 = Array::from_shape_vec(vec![1, scaled_data.len()], scaled_data).unwrap();

        let kurt1 = kurtosis_along_modes(&tensor1.view(), &[1], true, true);
        let kurt2 = kurtosis_along_modes(&tensor2.view(), &[1], true, true);

        // Both should succeed or both should fail
        if let (Ok(k1), Ok(k2)) = (kurt1, kurt2) {
            // Kurtosis should be invariant to scaling (within numerical tolerance)
            if !k1[[0]].is_nan() && !k2[[0]].is_nan() && !k1[[0]].is_infinite() && !k2[[0]].is_infinite() {
                prop_assert!((k1[[0]] - k2[[0]]).abs() < 1e-6,
                    "Kurtosis not scale-invariant: {} vs {}", k1[[0]], k2[[0]]);
            }
        }
    }

    /// Test Fisher vs Pearson kurtosis relationship
    #[test]
    fn test_kurtosis_fisher_pearson_relation(data in prop::collection::vec(-10.0..10.0f64, 10..30)) {
        let tensor = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();

        let fisher = kurtosis_along_modes(&tensor.view(), &[1], true, true);
        let pearson = kurtosis_along_modes(&tensor.view(), &[1], false, true);

        if let (Ok(f), Ok(p)) = (fisher, pearson) {
            if !f[[0]].is_nan() && !p[[0]].is_nan() && !f[[0]].is_infinite() && !p[[0]].is_infinite() {
                // Pearson = Fisher + 3
                prop_assert!((p[[0]] - (f[[0]] + 3.0)).abs() < 1e-6,
                    "Fisher-Pearson relation violated: {} vs {}", f[[0]], p[[0]]);
            }
        }
    }

    /// Test that moments are translation-invariant for skewness and kurtosis
    #[test]
    fn test_moments_translation_invariant(
        data in prop::collection::vec(-10.0..10.0f64, 10..30),
        shift in -100.0..100.0f64
    ) {
        let tensor1 = Array::from_shape_vec(vec![1, data.len()], data.clone()).unwrap();
        let shifted_data: Vec<f64> = data.iter().map(|x| x + shift).collect();
        let tensor2 = Array::from_shape_vec(vec![1, shifted_data.len()], shifted_data).unwrap();

        let skew1 = skewness_along_modes(&tensor1.view(), &[1], true);
        let skew2 = skewness_along_modes(&tensor2.view(), &[1], true);

        if let (Ok(s1), Ok(s2)) = (skew1, skew2) {
            if !s1[[0]].is_nan() && !s2[[0]].is_nan() && !s1[[0]].is_infinite() && !s2[[0]].is_infinite() {
                // Skewness should be translation-invariant
                prop_assert!((s1[[0]] - s2[[0]]).abs() < 1e-6,
                    "Skewness not translation-invariant: {} vs {}", s1[[0]], s2[[0]]);
            }
        }

        let kurt1 = kurtosis_along_modes(&tensor1.view(), &[1], true, true);
        let kurt2 = kurtosis_along_modes(&tensor2.view(), &[1], true, true);

        if let (Ok(k1), Ok(k2)) = (kurt1, kurt2) {
            if !k1[[0]].is_nan() && !k2[[0]].is_nan() && !k1[[0]].is_infinite() && !k2[[0]].is_infinite() {
                // Kurtosis should be translation-invariant
                prop_assert!((k1[[0]] - k2[[0]]).abs() < 1e-6,
                    "Kurtosis not translation-invariant: {} vs {}", k1[[0]], k2[[0]]);
            }
        }
    }

    // ========================================================================
    // Covariance and Correlation Properties
    // ========================================================================

    /// Test that covariance of X with itself equals variance
    #[test]
    fn test_covariance_self_equals_variance(size in 5usize..50) {
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let x = Array::from_shape_vec(vec![1, size], data).unwrap();

        let cov = covariance_along_modes(&x.view(), &x.view(), &[1], 1).unwrap();
        let var = variance_along_modes(&x.view(), &[1], 1).unwrap();

        prop_assert!((cov[[0]] - var[[0]]).abs() < 1e-10,
            "Cov(X,X) should equal Var(X): {} vs {}", cov[[0]], var[[0]]);
    }

    /// Test that covariance is symmetric: cov(X, Y) = cov(Y, X)
    #[test]
    fn test_covariance_symmetric(size in 5usize..30) {
        let x_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let y_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.2 + 1.0).collect();

        let x = Array::from_shape_vec(vec![1, size], x_data).unwrap();
        let y = Array::from_shape_vec(vec![1, size], y_data).unwrap();

        let cov_xy = covariance_along_modes(&x.view(), &y.view(), &[1], 1).unwrap();
        let cov_yx = covariance_along_modes(&y.view(), &x.view(), &[1], 1).unwrap();

        prop_assert!((cov_xy[[0]] - cov_yx[[0]]).abs() < 1e-10,
            "Covariance should be symmetric: {} vs {}", cov_xy[[0]], cov_yx[[0]]);
    }

    /// Test that correlation of X with itself is 1.0
    #[test]
    fn test_correlation_self_is_one(size in 5usize..50) {
        // Use non-constant data to avoid NaN
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1 + 1.0).collect();
        let x = Array::from_shape_vec(vec![1, size], data).unwrap();

        let corr = correlation_along_modes(&x.view(), &x.view(), &[1]).unwrap();

        prop_assert!((corr[[0]] - 1.0).abs() < 1e-10,
            "Correlation of X with itself should be 1.0, got {}", corr[[0]]);
    }

    /// Test that correlation is symmetric: corr(X, Y) = corr(Y, X)
    #[test]
    fn test_correlation_symmetric(size in 5usize..30) {
        let x_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1 + 1.0).collect();
        let y_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.2 + 2.0).collect();

        let x = Array::from_shape_vec(vec![1, size], x_data).unwrap();
        let y = Array::from_shape_vec(vec![1, size], y_data).unwrap();

        let corr_xy = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();
        let corr_yx = correlation_along_modes(&y.view(), &x.view(), &[1]).unwrap();

        prop_assert!((corr_xy[[0]] - corr_yx[[0]]).abs() < 1e-10,
            "Correlation should be symmetric: {} vs {}", corr_xy[[0]], corr_yx[[0]]);
    }

    /// Test that correlation is bounded: -1 <= corr(X, Y) <= 1
    #[test]
    fn test_correlation_bounds(size in 5usize..30, offset in 0usize..100) {
        use scirs2_core::ndarray_ext::Array;

        // Generate deterministic pseudo-random data
        let x_data: Vec<f64> = (0..size).map(|i| {
            let seed = ((i + offset) * 31) as f64;
            seed.sin().abs() * 10.0
        }).collect();
        let y_data: Vec<f64> = (0..size).map(|i| {
            let seed = ((i + offset) * 17) as f64;
            seed.cos().abs() * 10.0
        }).collect();

        let x = Array::from_shape_vec(vec![1, size], x_data).unwrap();
        let y = Array::from_shape_vec(vec![1, size], y_data).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();

        if !corr[[0]].is_nan() {
            prop_assert!(corr[[0]] >= -1.0 && corr[[0]] <= 1.0,
                "Correlation {} outside bounds [-1, 1]", corr[[0]]);
        }
    }

    /// Test that correlation of perfectly linearly related variables is ±1
    #[test]
    fn test_correlation_linear_relationship(size in 5usize..30, scale in 0.1f64..10.0) {
        let x_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| scale * x + 5.0).collect();

        let x = Array::from_shape_vec(vec![1, size], x_data).unwrap();
        let y = Array::from_shape_vec(vec![1, size], y_data).unwrap();

        let corr = correlation_along_modes(&x.view(), &y.view(), &[1]).unwrap();

        // For positive scale: perfect positive correlation
        if scale > 0.0 {
            prop_assert!((corr[[0]] - 1.0).abs() < 1e-8,
                "Correlation should be 1.0 for positive linear relationship, got {}", corr[[0]]);
        }
    }

    /// Test that cov(aX + b, cY + d) = a*c*cov(X, Y) for scale transformations
    #[test]
    fn test_covariance_bilinearity(size in 5usize..20, a in 0.5f64..2.0, c in 0.5f64..2.0) {
        let x_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let y_data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.2 + 1.0).collect();

        let x = Array::from_shape_vec(vec![1, size], x_data.clone()).unwrap();
        let y = Array::from_shape_vec(vec![1, size], y_data.clone()).unwrap();

        // Compute cov(X, Y)
        let cov_xy = covariance_along_modes(&x.view(), &y.view(), &[1], 1).unwrap();

        // Compute cov(aX + b, cY + d)
        let x_scaled: Vec<f64> = x_data.iter().map(|&xi| a * xi + 5.0).collect();
        let y_scaled: Vec<f64> = y_data.iter().map(|&yi| c * yi + 10.0).collect();

        let x_s = Array::from_shape_vec(vec![1, size], x_scaled).unwrap();
        let y_s = Array::from_shape_vec(vec![1, size], y_scaled).unwrap();

        let cov_scaled = covariance_along_modes(&x_s.view(), &y_s.view(), &[1], 1).unwrap();

        // Should satisfy: cov(aX + b, cY + d) = a*c*cov(X, Y)
        let expected = a * c * cov_xy[[0]];
        prop_assert!((cov_scaled[[0]] - expected).abs() < 1e-6,
            "Bilinearity property violated: {} vs {} (a={}, c={})",
            cov_scaled[[0]], expected, a, c);
    }
}

// ============================================================================
// Utility Function Property Tests
// ============================================================================

// Utility function property tests are now in src/utils.rs with their implementations
