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
// Utility Function Property Tests
// ============================================================================

// Utility function property tests are now in src/utils.rs with their implementations
